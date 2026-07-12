"""
Handwritten math layout engine (a tiny TeX).

Renders a LaTeX-math subset as a single image whose baseline metrics are known,
compositing:
  - the user's REAL handwritten glyphs for letters and digits (a-z, A-Z, 0-9)
  - math symbols (int, sum, sqrt, Greek, operators) rasterized from matplotlib
    mathtext, recolored to the same ink and lightly hand-jittered so they blend

Supports: juxtaposition, ^ and _ scripts, \\frac, \\sqrt, big operators
(\\int \\sum \\prod \\oint) with limits, \\left..\\right delimiters, \\mathrm/\\text,
Greek letters and common operators. Unknown commands fall back to a mathtext
symbol so nothing crashes.

Box model: each Box carries a float alpha (H,W in [0,1]) plus ascent/descent in
pixels measured from the baseline. Everything composes bottom-up.
"""
import numpy as np
from PIL import Image
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from scipy.ndimage import (gaussian_filter, map_coordinates, grey_dilation,
                           grey_erosion, distance_transform_edt)

from .metrics import char_box

_PARSER = MathTextParser("agg")
_XH_RATIO = 0.755          # mathtext x-height / fontsize at dpi 100
_DPI = 100

# latex command -> mathtext token to rasterize
SYMBOLS = {
    r"\int": r"\int", r"\iint": r"\iint", r"\oint": r"\oint",
    r"\sum": r"\sum", r"\prod": r"\prod", r"\lim": r"\lim",
    r"\infty": r"\infty", r"\partial": r"\partial", r"\nabla": r"\nabla",
    r"\pm": r"\pm", r"\mp": r"\mp", r"\times": r"\times", r"\div": r"\div",
    r"\cdot": r"\cdot", r"\ast": r"\ast", r"\star": r"\star",
    r"\leq": r"\leq", r"\geq": r"\geq", r"\neq": r"\neq", r"\approx": r"\approx",
    r"\equiv": r"\equiv", r"\propto": r"\propto", r"\sim": r"\sim",
    r"\to": r"\to", r"\rightarrow": r"\rightarrow", r"\leftarrow": r"\leftarrow",
    r"\Rightarrow": r"\Rightarrow", r"\mapsto": r"\mapsto",
    r"\in": r"\in", r"\notin": r"\notin", r"\subset": r"\subset",
    r"\cup": r"\cup", r"\cap": r"\cap", r"\forall": r"\forall",
    r"\exists": r"\exists", r"\angle": r"\angle", r"\perp": r"\perp",
    r"\cdots": r"\cdots", r"\ldots": r"\ldots", r"\dots": r"\ldots",
    # greek
    r"\alpha": r"\alpha", r"\beta": r"\beta", r"\gamma": r"\gamma",
    r"\delta": r"\delta", r"\epsilon": r"\epsilon", r"\varepsilon": r"\varepsilon",
    r"\zeta": r"\zeta", r"\eta": r"\eta", r"\theta": r"\theta",
    r"\iota": r"\iota", r"\kappa": r"\kappa", r"\lambda": r"\lambda",
    r"\mu": r"\mu", r"\nu": r"\nu", r"\xi": r"\xi", r"\pi": r"\pi",
    r"\rho": r"\rho", r"\sigma": r"\sigma", r"\tau": r"\tau",
    r"\phi": r"\phi", r"\varphi": r"\varphi", r"\chi": r"\chi",
    r"\psi": r"\psi", r"\omega": r"\omega",
    r"\Gamma": r"\Gamma", r"\Delta": r"\Delta", r"\Theta": r"\Theta",
    r"\Lambda": r"\Lambda", r"\Sigma": r"\Sigma", r"\Phi": r"\Phi",
    r"\Omega": r"\Omega", r"\Pi": r"\Pi",
}
BIG_OPS = {r"\int", r"\iint", r"\oint", r"\sum", r"\prod", r"\lim"}
STACK_LIMITS = {r"\sum", r"\prod", r"\lim"}   # limits go above/below in display
OP_CHARS = set("+-=<>*/|,.:;!()[]")


class Box:
    __slots__ = ("alpha", "ascent", "descent", "_lspace", "_stack_limits")

    def __init__(self, alpha, ascent, descent):
        self.alpha = alpha                       # float32 (H,W) in [0,1]
        self.ascent = float(ascent)
        self.descent = float(descent)
        self._lspace = 0.05
        self._stack_limits = False

    @property
    def w(self):
        return self.alpha.shape[1]

    @property
    def h(self):
        return self.alpha.shape[0]


def _empty(w=1, h=1):
    return Box(np.zeros((max(1, int(h)), max(1, int(w))), np.float32), h, 0)


def _disk(r):
    r = int(max(1, round(r)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r)


def _stroke_px(alpha, thr=0.5):
    """Estimate stroke width in px = 2 x median distance-to-edge on the ink."""
    ink = alpha > thr
    if ink.sum() < 4:
        return 1.0
    d = distance_transform_edt(ink)
    return float(2.0 * np.median(d[ink]))


def _elastic(alpha, sigma, amp, rng):
    """Smooth random displacement field -> organic hand tremor / wobble."""
    h, w = alpha.shape
    if h < 3 or w < 3 or amp <= 0:
        return alpha
    dx = gaussian_filter(rng.randn(h, w), sigma) * amp
    dy = gaussian_filter(rng.randn(h, w), sigma) * amp
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    warped = map_coordinates(alpha, [(yy + dy).ravel(), (xx + dx).ravel()],
                             order=1, mode="constant").reshape(h, w)
    return warped.astype(np.float32)


def _restyle(a, S, rng, style, strength):
    """Core: thicken to the user's stroke weight, add tremor, edge roughness and
    uneven ink density. Operates in place on a padded array (same shape out)."""
    # 1) match the user's stroke weight (thicken thin vector strokes toward it)
    target = style["stroke_ratio"] * S
    cur = _stroke_px(a)
    delta = (target - cur) * strength
    if delta > 0.6:
        a = grey_dilation(a, footprint=_disk(min(delta / 2.0, 0.14 * S)))
    elif delta < -0.6:
        a = grey_erosion(a, footprint=_disk(min(-delta / 2.0, 0.06 * S)))
    # 2) low-frequency tremor (long gentle waves)
    a = _elastic(a, sigma=0.42 * S, amp=style["tremor"] * S * strength, rng=rng)
    # 3) fine edge roughness (short wavelength, small amplitude)
    a = _elastic(a, sigma=max(1.3, 0.05 * S),
                 amp=style["rough"] * S * strength, rng=rng)
    # 4) uneven ink density (pressure) + faint dry-pen speckle
    field = 1.0 + 0.16 * strength * gaussian_filter(rng.randn(*a.shape), 0.5 * S)
    a = np.clip(a * field, 0, 1)
    speck = gaussian_filter(rng.randn(*a.shape), 1.0)
    a = np.clip(a - 0.10 * strength * (speck > 1.4), 0, 1)
    return a


def handwritify(alpha, ascent, S, rng, style, strength=1.0):
    """Treat the clean symbol alpha as a PRIOR shape and restyle it with the
    user's measured hand characteristics, freshly sampled each call. Returns
    (new_alpha, new_ascent) with the baseline tracked through the transform."""
    if alpha.size == 0 or alpha.max() <= 0 or strength <= 0:
        return alpha, ascent
    pad = int(max(3, 0.25 * S))
    a = np.pad(alpha.astype(np.float32), pad)
    a = _restyle(a, S, rng, style, strength)
    ys, xs = np.where(a > 0.08)
    if len(xs) == 0:
        return alpha, ascent
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    new_ascent = (ascent + pad) - y0
    return a[y0:y1, x0:x1], new_ascent


def restyle_stroke(alpha, S, rng, style, strength=1.0):
    """Restyle a drawn stroke (fraction bar, radical) keeping its canvas shape."""
    if alpha.max() <= 0 or strength <= 0:
        return alpha
    pad = int(max(2, 0.12 * S))
    a = _restyle(np.pad(alpha.astype(np.float32), pad), S, rng, style, strength)
    return a[pad:pad + alpha.shape[0], pad:pad + alpha.shape[1]]


def _blit(canvas, alpha, top, left):
    """Composite alpha onto canvas at (top,left) with clipping (max blend)."""
    H, W = canvas.shape
    h, w = alpha.shape
    top, left = int(top), int(left)
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(H, top + h), min(W, left + w)
    if y1 <= y0 or x1 <= x0:
        return
    sub = alpha[y0 - top:y1 - top, x0 - left:x1 - left]
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], sub)


# ---------------------------------------------------------------------------
class MathHand:
    def __init__(self, renderer, ink=(20, 24, 60), rng=None, style_strength=1.0):
        self.r = renderer                        # HandwritingRenderer (glyph bank)
        self.ink = ink
        self.rng = rng or np.random.RandomState(0)
        self.style_strength = style_strength
        self.style = self._measure_style()

    def _measure_style(self):
        """Style used to restyle symbols so they match the user's hand. Weight is
        tied to the SAME target as the regularized letters, so symbols and glyphs
        share a stroke weight; tremor/roughness give the hand-drawn finish."""
        stroke_ratio = getattr(self.r, "stroke_ratio", 0.11)
        return {"stroke_ratio": stroke_ratio, "tremor": 0.032, "rough": 0.017}

    # ---- leaf: real handwritten glyph -------------------------------------
    def glyph(self, ch, S):
        mask = self.r._resolve(ch)
        if mask is None:
            return self.symbol(ch, S)            # fall back to typeset shape
        bottom, top = char_box(ch)
        h = max(2, int(round((top - bottom) * S)))
        w = max(1, int(round(mask.shape[1] * h / mask.shape[0])))
        g = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                       .resize((w, h), Image.LANCZOS), np.float32) / 255.0
        dy = 0.0
        reg = getattr(self.r, "regularize", 0.0)
        if reg > 0:
            from .imageops import normalize_stroke
            g = np.pad(g, 5)
            g = normalize_stroke(g, getattr(self.r, "stroke_ratio", 0.11) * S, reg)
            ys, xs = np.where(g > 0.12)
            if len(xs):
                g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                dy = (g.shape[0] - h) / 2.0
        return Box(g, top * S + dy, -bottom * S + dy)

    # ---- leaf: typeset symbol in matching ink -----------------------------
    def symbol(self, latex, S, jitter=True):
        F = S / _XH_RATIO
        try:
            rp = _PARSER.parse(f"${latex}$", dpi=_DPI, prop=FontProperties(size=F))
            a = np.asarray(rp.image, np.float32) / 255.0
            depth = rp.depth
        except Exception:
            return _empty(int(0.3 * S), int(S))
        if a.ndim == 0 or a.size == 0:
            return _empty(int(0.3 * S), int(S))
        ascent = a.shape[0] - depth
        if jitter and a.shape[0] > 3 and a.shape[1] > 3:
            a, ascent = handwritify(a, ascent, S, self.rng, self.style,
                                    self.style_strength)
            depth = a.shape[0] - ascent
        return Box(a, ascent, depth)

    # ---- combinators ------------------------------------------------------
    def hcat(self, boxes, gap=0.0):
        boxes = [b for b in boxes if b is not None]
        if not boxes:
            return _empty()
        asc = max(b.ascent for b in boxes)
        gpx = int(round(gap))
        W = sum(b.w for b in boxes) + gpx * (len(boxes) - 1)
        # derive height from actual integer placements to avoid rounding overflow
        tops = [int(round(asc - b.ascent)) for b in boxes]
        H = max(t + b.h for t, b in zip(tops, boxes))
        top_min = min(tops)
        if top_min < 0:                          # a box rises above asc; shift down
            tops = [t - top_min for t in tops]
            H -= top_min
            asc -= top_min
        canvas = np.zeros((max(1, H), max(1, W)), np.float32)
        x = 0
        for top, b in zip(tops, boxes):
            canvas[top:top + b.h, x:x + b.w] = np.maximum(
                canvas[top:top + b.h, x:x + b.w], b.alpha)
            x += b.w + gpx
        return Box(canvas, asc, H - asc)

    def _stack(self, top_box, bot_box, align="center"):
        """Vertically stack two boxes; return canvas + the y of their split."""
        W = max(top_box.w, bot_box.w)
        H = top_box.h + bot_box.h
        canvas = np.zeros((H, W), np.float32)
        for b, y in [(top_box, 0), (bot_box, top_box.h)]:
            ox = (W - b.w) // 2
            canvas[y:y + b.h, ox:ox + b.w] = np.maximum(
                canvas[y:y + b.h, ox:ox + b.w], b.alpha)
        return canvas

    def frac(self, num, den, S):
        W = int(max(num.w, den.w) + 0.5 * S)
        bar_t = max(2, int(0.06 * S))
        gap = int(0.18 * S)
        axis = 0.30 * S                          # bar sits on the math axis
        canvas_h = num.h + gap + bar_t + gap + den.h
        canvas = np.zeros((canvas_h, W), np.float32)
        nx = (W - num.w) // 2
        canvas[0:num.h, nx:nx + num.w] = num.alpha
        by = num.h + gap
        band = int(0.6 * S)
        strip = np.zeros((band, W), np.float32)
        strip[band // 2:band // 2 + bar_t, :] = 1.0
        strip = restyle_stroke(strip, S, self.rng, self.style, self.style_strength)
        _blit(canvas, strip, by - band // 2 + bar_t // 2, 0)
        dy = by + bar_t + gap
        dx = (W - den.w) // 2
        canvas[dy:dy + den.h, dx:dx + den.w] = den.alpha
        ascent = by + bar_t / 2 + axis
        descent = canvas_h - ascent
        return Box(canvas, ascent, descent)

    def sqrt(self, content, S):
        pad = int(0.12 * S)
        bar_t = max(2, int(0.05 * S))
        over = int(0.16 * S)                     # space above content for vinculum
        rad_w = int(0.55 * S)
        c = content
        H = int(c.ascent + c.descent + over + bar_t)
        W = c.w + rad_w + pad * 2
        top = over + bar_t
        # radical + vinculum on their own layer, then restyle only those strokes
        rad = Image.new("L", (W, H), 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(rad)
        d.line([(rad_w, bar_t // 2), (W, bar_t // 2)], fill=255, width=bar_t)
        x0, y0 = int(rad_w * 0.15), int(H * 0.60)
        x1, y1 = int(rad_w * 0.42), H - bar_t
        x2, y2 = rad_w, bar_t // 2
        d.line([(x0, y0), (x1, y1)], fill=255, width=bar_t)
        d.line([(x1, y1), (x2, y2)], fill=255, width=bar_t)
        rad = restyle_stroke(np.asarray(rad, np.float32) / 255.0, S,
                             self.rng, self.style, self.style_strength)
        canvas = np.zeros((H, W), np.float32)
        canvas[:rad.shape[0], :rad.shape[1]] = rad[:H, :W]
        _blit(canvas, c.alpha, top, rad_w + pad)
        ascent = top + c.ascent
        return Box(canvas, ascent, H - ascent)

    def scripts(self, base, sup, sub, S, is_big=False):
        if is_big and (sup or sub) and getattr(base, "_stack_limits", False):
            return self._stacked_limits(base, sup, sub, S)
        ss = 0.62 * S
        parts = [base]
        # build a small column for sup/sub to the right
        col_boxes = []
        sup_b = sup
        sub_b = sub
        raise_px = 0.5 * base.ascent
        drop_px = 0.42 * base.descent + 0.3 * S
        b = base
        W = b.w + int(0.02 * S) + max((sup_b.w if sup_b else 0),
                                      (sub_b.w if sub_b else 0))
        top_extra = (sup_b.ascent + raise_px - b.ascent) if sup_b else 0
        bot_extra = (sub_b.descent + drop_px - b.descent) if sub_b else 0
        asc = b.ascent + max(0, top_extra)
        desc = b.descent + max(0, bot_extra)
        H = int(round(asc + desc))
        canvas = np.zeros((max(1, H), max(1, W)), np.float32)
        baseline = asc
        _blit(canvas, b.alpha, round(baseline - b.ascent), 0)
        sx = b.w + int(0.02 * S)
        if sup_b:
            _blit(canvas, sup_b.alpha, round(baseline - raise_px - sup_b.ascent), sx)
        if sub_b:
            _blit(canvas, sub_b.alpha, round(baseline + drop_px - sub_b.ascent), sx)
        return Box(canvas, asc, desc)

    def _stacked_limits(self, base, sup, sub, S):
        cols = [c for c in [sup, base, sub] if c is not None]
        W = max(c.w for c in cols)
        gap = int(0.06 * S)
        parts = []
        if sup:
            parts.append(sup)
        parts.append(base)
        if sub:
            parts.append(sub)
        H = sum(c.h for c in parts) + gap * (len(parts) - 1)
        canvas = np.zeros((H, W), np.float32)
        y = 0
        base_top = 0
        for c in parts:
            ox = (W - c.w) // 2
            canvas[y:y + c.h, ox:ox + c.w] = np.maximum(
                canvas[y:y + c.h, ox:ox + c.w], c.alpha)
            if c is base:
                base_top = y
            y += c.h + gap
        ascent = base_top + base.ascent
        return Box(canvas, ascent, H - ascent)

    # ---- colorize ---------------------------------------------------------
    def to_image(self, box):
        h, w = box.alpha.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = self.ink
        rgba[..., 3] = (np.clip(box.alpha, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(rgba, "RGBA")

    def spacer(self, w):
        return Box(np.zeros((1, max(1, int(w))), np.float32), 0, 0)

    def delim(self, ch, target_h, S):
        b = self.symbol(ch, S, jitter=False)
        if b.h < 2:
            return b
        scale = max(1.0, target_h / b.h)
        nh = int(b.h * scale)
        nw = max(1, int(b.w * min(scale, 1.6)))
        a = np.asarray(Image.fromarray((b.alpha * 255).astype(np.uint8))
                       .resize((nw, nh), Image.LANCZOS), np.float32) / 255.0
        return Box(a, nh / 2 + 0.25 * S, nh / 2 - 0.25 * S)


# ---------------------------------------------------------------------------
# Tokenizer + recursive-descent parser
# ---------------------------------------------------------------------------
def _tokenize(s):
    toks = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            j = i + 1
            if j < n and s[j].isalpha():
                k = j
                while k < n and s[k].isalpha():
                    k += 1
                toks.append(("cmd", s[i:k]))
                i = k
            else:
                toks.append(("cmd", s[i:j + 1]))   # \{  \,  \|
                i = j + 1
        elif c in "{}":
            toks.append(("lbrace" if c == "{" else "rbrace", c))
            i += 1
        elif c == "^":
            toks.append(("sup", c)); i += 1
        elif c == "_":
            toks.append(("sub", c)); i += 1
        elif c.isspace():
            i += 1
        else:
            toks.append(("char", c)); i += 1
    return toks


class _Parser:
    def __init__(self, mh, S, display=False):
        self.mh = mh
        self.S = S
        self.display = display

    def parse(self, toks):
        box, i = self.sequence(toks, 0, self.S)
        return box

    def sequence(self, toks, i, S):
        boxes = []
        while i < len(toks) and toks[i][0] != "rbrace":
            atom, i = self.atom(toks, i, S)
            if atom is not None:
                boxes.append(atom)
        if not boxes:
            return _empty(1, int(S)), i
        return self._join(boxes, S), i

    def _join(self, boxes, S):
        out = []
        for k, b in enumerate(boxes):
            if k > 0:
                gap = getattr(b, "_lspace", 0.05) * S
                out.append(self.mh.spacer(gap))
            out.append(b)
        return self.mh.hcat(out, gap=0)

    def atom(self, toks, i, S):
        prim, i, is_big = self.primary(toks, i, S)
        if prim is None:
            return None, i
        sup = sub = None
        while i < len(toks) and toks[i][0] in ("sup", "sub"):
            kind = toks[i][0]
            i += 1
            s_box, i = self.primary_as_box(toks, i, 0.62 * S)
            if kind == "sup":
                sup = s_box
            else:
                sub = s_box
        if sup is not None or sub is not None:
            prim._stack_limits = is_big and self._display_limits
            return self.mh.scripts(prim, sup, sub, S, is_big=is_big), i
        return prim, i

    def primary_as_box(self, toks, i, S):
        if i < len(toks) and toks[i][0] == "lbrace":
            box, i = self.sequence(toks, i + 1, S)
            if i < len(toks) and toks[i][0] == "rbrace":
                i += 1
            return box, i
        prim, i, _ = self.primary(toks, i, S)
        return prim if prim is not None else _empty(1, int(S)), i

    def primary(self, toks, i, S):
        if i >= len(toks):
            return None, i, False
        kind, val = toks[i]
        if kind == "lbrace":
            box, i = self.sequence(toks, i + 1, S)
            if i < len(toks) and toks[i][0] == "rbrace":
                i += 1
            return box, i, False
        if kind == "char":
            i += 1
            b = self.mh.glyph(val, S) if (val.isalnum()) else self.mh.symbol(val, S)
            if val in OP_CHARS and val not in "()[]|.,":
                b._lspace = 0.18
            return b, i, False
        if kind == "cmd":
            return self.command(toks, i, S)
        i += 1
        return _empty(1, int(S)), i, False

    def command(self, toks, i, S):
        val = toks[i][1]
        i += 1
        if val == r"\frac":
            num, i = self.primary_as_box(toks, i, 0.78 * S)
            den, i = self.primary_as_box(toks, i, 0.78 * S)
            return self.mh.frac(num, den, S), i, False
        if val == r"\sqrt":
            if i < len(toks) and toks[i] == ("char", "["):
                while i < len(toks) and toks[i] != ("char", "]"):
                    i += 1
                i += 1
            content, i = self.primary_as_box(toks, i, S)
            return self.mh.sqrt(content, S), i, False
        if val in (r"\left", r"\right"):
            # \left<d> ... \right<d>
            delim_ch = ""
            if i < len(toks) and toks[i][0] == "char":
                delim_ch = toks[i][1]; i += 1
            elif i < len(toks) and toks[i][0] == "cmd":
                delim_ch = "|"; i += 1
            if val == r"\right":
                return _empty(1, 1), i, False        # handled by \left scan
            inner, i = self.until_right(toks, i, S)
            close = "("
            return self.wrap_delims(inner, delim_ch, S), i, False
        if val in (r"\mathrm", r"\text", r"\mathbf", r"\operatorname"):
            box, i = self.primary_as_box(toks, i, S)
            return box, i, False
        if val in SYMBOLS:
            is_big = val in BIG_OPS
            self._display_limits = (val in STACK_LIMITS) and self.display
            b = self.mh.symbol(SYMBOLS[val], S if not is_big else S * 1.15)
            if not is_big:
                b._lspace = 0.14
            return b, i, is_big
        if val in (r"\,", r"\;", r"\ ", r"\quad", r"\qquad"):
            w = {r"\,": 0.15, r"\;": 0.25, r"\ ": 0.3,
                 r"\quad": 0.6, r"\qquad": 1.1}[val] * S
            return self.mh.spacer(w), i, False
        if val == r"\\":
            return self.mh.spacer(0), i, False
        # unknown command: try mathtext
        return self.mh.symbol(val, S), i, False

    def until_right(self, toks, i, S):
        boxes = []
        depth = 0
        start = i
        while i < len(toks):
            if toks[i][0] == "cmd" and toks[i][1] == r"\left":
                depth += 1
            if toks[i][0] == "cmd" and toks[i][1] == r"\right":
                if depth == 0:
                    break
                depth -= 1
            i += 1
        inner_toks = toks[start:i]
        # consume \right<delim>
        if i < len(toks) and toks[i][1] == r"\right":
            i += 1
            if i < len(toks) and toks[i][0] in ("char", "cmd"):
                i += 1
        box, _ = self.sequence(inner_toks, 0, S)
        return box, i

    def wrap_delims(self, inner, open_ch, S):
        h = inner.ascent + inner.descent
        parts = []
        if open_ch and open_ch != ".":
            parts.append(self.mh.delim(open_ch, h, S))
        parts.append(inner)
        close = {"(": ")", "[": "]", "{": "}", "|": "|"}.get(open_ch, ")")
        parts.append(self.mh.delim(close, h, S))
        return self.mh.hcat(parts, gap=0)

    _display_limits = False


def render_math_hand(renderer, latex, S, ink=(20, 24, 60), rng=None,
                     display=False, style_strength=1.0):
    """Return an RGBA PIL image + (ascent_px, descent_px) baseline metrics."""
    mh = MathHand(renderer, ink=ink, rng=rng, style_strength=style_strength)
    p = _Parser(mh, S, display=display)
    box = p.parse(_tokenize(latex))
    return mh.to_image(box), box.ascent, box.descent

