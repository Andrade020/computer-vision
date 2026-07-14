"""
Handwriting renderer: text / rich documents -> page image(s).

Primary source is the real-ink glyph bank (best fidelity, natural variation
because a different real sample is picked per occurrence). A neural GlyphGenerator
can be plugged in as a fallback for missing / low-data characters.

A single flow engine lays out "units" (handwritten words, spaces, inline math
images, breaks) with word wrapping, so plain text and LaTeX documents share code.

Depends only on numpy + PIL (torch only if a model is supplied).
"""
import os
import pickle
import random
import numpy as np
from PIL import Image, ImageDraw

from .metrics import char_box, SPACE_ADVANCE

HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.path.join(HERE, "data", "glyph_bank.pkl")

_PUNCT_S = 40


def _tight(arr):
    ys, xs = np.where(arr > 0.08)
    if len(xs) == 0:
        return None
    return arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _synthetic_punct(ch):
    """Procedural fallback for common punctuation the OCR bank never captured
    (it only has letters/digits). Drawn once, tightly cropped, and then run
    through the same stroke-regularization + rotation-jitter pipeline as real
    glyphs in _glyph_atom, so it doesn't look stamped-on."""
    S = _PUNCT_S
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    r = int(0.09 * S)
    cx = S * 0.5
    if ch == ".":
        d.ellipse([cx - r, S * 0.63, cx + r, S * 0.63 + 2 * r], fill=255)
    elif ch == ",":
        cy = S * 0.63
        d.ellipse([cx - r, cy, cx + r, cy + 2 * r], fill=255)
        d.line([(cx, cy + 1.6 * r), (cx - 1.4 * r, cy + 4 * r)],
              fill=255, width=max(2, int(0.05 * S)))
    elif ch in (":", ";"):
        r2 = int(0.075 * S)
        d.ellipse([cx - r2, S * 0.15, cx + r2, S * 0.15 + 2 * r2], fill=255)
        cy2 = S * 0.55
        d.ellipse([cx - r2, cy2, cx + r2, cy2 + 2 * r2], fill=255)
        if ch == ";":
            d.line([(cx, cy2 + 1.6 * r2), (cx - 1.4 * r2, cy2 + 4 * r2)],
                  fill=255, width=max(2, int(0.045 * S)))
    elif ch == "-":
        d.line([(S * 0.15, S * 0.5), (S * 0.85, S * 0.5)],
              fill=255, width=max(2, int(0.09 * S)))
    elif ch == "!":
        w = max(2, int(0.09 * S))
        d.line([(cx, S * 0.05), (cx, S * 0.55)], fill=255, width=w)
        d.ellipse([cx - r * 0.9, S * 0.68, cx + r * 0.9, S * 0.68 + 1.8 * r],
                 fill=255)
    else:
        return None
    return _tight(np.asarray(im, np.float32) / 255.0)


class HandwritingRenderer:
    def __init__(self, bank_path=BANK_PATH, model=None, seed=None, hand_math=False,
                 math_style=1.0, regularize=1.0, stroke_ratio=0.11):
        with open(bank_path, "rb") as f:
            d = pickle.load(f)
        self.bank = d["bank"]
        self.classes = set(d["classes"])
        self.model = model
        self.hand_math = hand_math          # render math in the user's hand
        self.math_style = math_style        # 0=clean symbols .. 1=default .. more=rougher
        self.regularize = regularize        # 0=raw glyph weight .. 1=fully normalized
        self.stroke_ratio = stroke_ratio    # target stroke width as fraction of x-height
        self.rng = random.Random(seed)
        self._npr = np.random.RandomState(seed if seed is not None else 0)

    # ---- glyph acquisition -------------------------------------------------
    def _bank_glyph(self, ch):
        s = self.bank.get(ch)
        if s:
            return s[self.rng.randrange(len(s))].astype(np.float32) / 255.0
        return None

    def _resolve(self, ch):
        g = self._bank_glyph(ch)
        if g is not None:
            return g
        alt = ch.swapcase()
        if alt != ch:
            g = self._bank_glyph(alt)
            if g is not None:
                return g
        if self.model is not None:
            key = ch if ch in self.classes else (
                alt if alt in self.classes else None)
            if key is not None:
                try:
                    return self.model.generate(key, self._npr)
                except Exception:
                    pass
        return _synthetic_punct(ch)

    def _final_glyph(self, mask, w_px, h_px, xh):
        """Resize a glyph to its render box and normalize its stroke weight so
        all glyphs share a consistent thickness. Returns (mask, w, h, dy) where
        dy is the vertical growth to fold into the baseline offset."""
        g = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                       .resize((w_px, h_px), Image.LANCZOS), np.float32) / 255.0
        if self.regularize <= 0:
            return g, w_px, h_px, 0
        from .imageops import normalize_stroke
        pad = 6
        g = np.pad(g, pad)
        g = normalize_stroke(g, self.stroke_ratio * xh, self.regularize)
        ys, xs = np.where(g > 0.12)
        if len(xs) == 0:
            return g[pad:pad + h_px, pad:pad + w_px], w_px, h_px, 0
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h2, w2 = g.shape
        return g, w2, h2, (h2 - h_px) / 2.0

    # ---- unit builders -----------------------------------------------------
    def _glyph_atom(self, ch, xh, emph=0.0):
        """Return dict with mask + geometry relative to baseline, or None."""
        mask = self._resolve(ch)
        if mask is None:
            return None
        bottom, top = char_box(ch)
        h_px = max(1, int(round((top - bottom) * xh)))
        scale = h_px / mask.shape[0]
        w_px = max(1, int(round(mask.shape[1] * scale)))
        mask, w_px, h_px, dy = self._final_glyph(mask, w_px, h_px, xh)
        top_px = int(round(top * xh + dy))     # above baseline (+ stroke growth)
        rot = float(self._npr.normal(0, 1.5))
        return {"mask": mask, "w": w_px, "h": h_px, "top": top_px, "rot": rot}

    def _word_unit(self, word, xh):
        atoms = []
        w = 0
        for ch in word:
            a = self._glyph_atom(ch, xh)
            if a is None:
                w += int(0.5 * xh)
                continue
            atoms.append((a, w))
            w += a["w"] + int(0.06 * xh)
        return {"kind": "glyphs", "atoms": atoms, "w": w}

    def _space_unit(self, xh):
        return {"kind": "space",
                "w": int(SPACE_ADVANCE * xh * (1.0 + 0.15 * self.rng.random()))}

    def _dot_unit(self, xh, ink):
        d = max(5, int(0.28 * xh))
        dot = Image.new("RGBA", (d, d), ink + (0,))
        dd = ImageDraw.Draw(dot)
        dd.ellipse([0, 0, d - 1, d - 1], fill=ink + (255,))
        u = self._image_unit(dot, ascender_frac=0.55)
        return u

    def _image_unit(self, img, ascender_frac=0.72):
        """Inline image (e.g. math); ascender_frac = fraction of height above baseline."""
        return {"kind": "image", "img": img, "w": img.width,
                "h": img.height, "asc": int(img.height * ascender_frac)}

    def _text_to_units(self, text, xh):
        units = []
        for i, word in enumerate(text.split(" ")):
            if i > 0:
                units.append(self._space_unit(xh))
            if word:
                units.append(self._word_unit(word, xh))
        return units

    # ---- flow engine -------------------------------------------------------
    def iter_document(self, blocks, xh=26, page_w=1000, margin=70,
                      ink=(20, 24, 60), bg=(252, 250, 244), ruled=False,
                      slant=0.0, on_block=None, on_error=None):
        """Yield finished PIL pages one at a time as they fill, instead of
        holding the whole document in memory. This is the "lazy" entry point:
        callers (see the CLIs) save each page to disk the moment it arrives,
        so a long document that fails partway through still leaves every
        completed page on disk instead of losing everything.

        Each block is wrapped in try/except: a single malformed construct
        (reported via on_error(index, block, exc) if given) just ends the
        current line and moves on, instead of aborting the whole document.
        on_block(index, total, block), if given, fires before each block --
        use it to print progress on a long run.
        """
        line_h = int((1.42 + 0.42 + 1.0) * xh)
        max_x = page_w - margin
        page_h = int(11.0 / 8.5 * page_w)          # letter aspect
        ready = []

        def new_page():
            img = Image.new("RGB", (page_w, page_h), bg)
            if ruled:
                self._draw_rules(img, margin, line_h, xh)
            return img

        img = new_page()
        y = margin + int(1.42 * xh)
        x = margin

        def advance_line(cur_y):
            nonlocal img
            ny = cur_y + line_h
            if ny > page_h - margin:
                ready.append(img)
                img = new_page()
                return margin + int(1.42 * xh)
            return ny

        total = len(blocks)
        for bi, blk in enumerate(blocks):
            if on_block:
                on_block(bi, total, blk)
            try:
                kind = blk["type"]
                if kind == "vspace":
                    y = min(page_h - margin, y + blk["px"])
                    x = margin
                elif kind == "rule":
                    x = margin
                    ry = y - int(0.35 * xh)
                    ImageDraw.Draw(img).line(
                        [(margin, ry), (page_w - margin, ry)],
                        fill=(190, 196, 212), width=2)
                    y = min(page_h - margin, y + int(blk.get("gap", 0.25) * line_h))
                else:
                    bxh = int(xh * blk.get("scale", 1.0))
                    indent = blk.get("indent", 0) * int(1.4 * xh)
                    x = margin + indent
                    if blk.get("newline_before") and x != margin:
                        y = advance_line(y)

                    units = self._blk_units(blk, bxh)
                    if blk.get("bullet_text"):
                        units = ([self._word_unit(blk["bullet_text"], bxh),
                                 self._space_unit(bxh)] + units)
                    elif blk.get("bullet"):
                        units = [self._dot_unit(bxh, ink), self._space_unit(bxh)] + units

                    if blk.get("center"):
                        total_w = sum(u["w"] for u in units)
                        x = max(margin + indent,
                               margin + (max_x - margin - total_w) // 2)

                    avail = max_x - margin - indent
                    for u in units:
                        if u["kind"] == "break":
                            x = margin + indent
                            y = advance_line(y)
                            continue
                        if u["kind"] == "image" and u["w"] > avail > 0:
                            # equation wider than the page -- shrink to fit
                            # instead of running off the edge with no wrap point
                            scale = avail / u["w"]
                            new_w, new_h = avail, max(1, int(u["h"] * scale))
                            u = {**u, "img": u["img"].resize((new_w, new_h), Image.LANCZOS),
                                "w": new_w, "h": new_h, "asc": int(u["asc"] * scale)}
                        if x + u["w"] > max_x and x > margin:
                            x = margin
                            y = advance_line(y)
                        if u["kind"] == "space":
                            x += u["w"]
                            continue
                        if u["kind"] == "glyphs":
                            self._paste_word(img, u, x, y, bxh, ink, slant)
                        elif u["kind"] == "image":
                            top = y - u["asc"]
                            img.paste(u["img"], (int(x), int(top)), u["img"])
                        x += u["w"]
                    y = advance_line(y)
                    y = min(page_h - margin, y + int(blk.get("gap", 0.2) * line_h))
                    x = margin
            except Exception as exc:
                if on_error:
                    on_error(bi, blk, exc)
                x = margin
                y = advance_line(y)
            while ready:
                yield ready.pop(0)

        yield img

    def render_document(self, blocks, xh=26, page_w=1000, margin=70,
                        ink=(20, 24, 60), bg=(252, 250, 244), ruled=False,
                        slant=0.0):
        """Rich blocks -> list of PIL pages (letter aspect). For long documents
        prefer iter_document directly so pages can be saved as they're produced."""
        return list(self.iter_document(blocks, xh, page_w, margin, ink, bg,
                                       ruled, slant))

    def _blk_units(self, blk, xh):
        units = []
        for run in blk["runs"]:
            if run[0] == "t":
                if units:
                    units.append(self._space_unit(xh))
                units += self._text_to_units(run[1], xh)
            elif run[0] == "m":
                ink = blk.get("ink", (20, 24, 60))
                if units:
                    units.append(self._space_unit(xh))
                units.append(self._math_unit(run[1], run[2], xh, ink))
            elif run[0] == "br":
                units.append({"kind": "break", "w": 0})
        return units

    def _math_unit(self, expr, display, xh, ink):
        """Build an inline-image unit for a math expression, baseline-aligned."""
        if self.hand_math:
            try:
                from .mathhand import render_math_hand
                S = int((1.25 if display else 1.05) * xh)
                mimg, asc, desc = render_math_hand(self, expr, S, ink=ink,
                                                   rng=self._npr, display=display,
                                                   style_strength=self.math_style)
                frac = asc / max(1.0, asc + desc)
                return self._image_unit(mimg, ascender_frac=frac)
            except Exception:
                pass                              # fall back to typeset
        from .mathimg import render_math
        tall = any(k in expr for k in ("\\frac", "\\sum", "\\int", "^", "_"))
        base = (1.9 if display else (1.55 if tall else 1.2))
        mimg, _ = render_math(expr, int(base * xh), ink=ink)
        return self._image_unit(mimg)

    def _paste_word(self, img, unit, x0, baseline_y, xh, ink, slant):
        for a, off in unit["atoms"]:
            mask, w, h, top = a["mask"], a["w"], a["h"], a["top"]
            jy = int(self._npr.normal(0, 0.6))
            slant_dx = int(-slant * top)
            gx = x0 + off + slant_dx
            gy = baseline_y - top + jy
            alpha = Image.fromarray((np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    (w, h), Image.LANCZOS))).astype(np.uint8))
            glyph = Image.new("RGBA", (w, h), ink + (0,))
            glyph.putalpha(alpha)
            if abs(a["rot"]) > 0.05:
                glyph = glyph.rotate(a["rot"], expand=True, resample=Image.BICUBIC)
            img.paste(glyph, (int(gx), int(gy)), glyph)

    def _draw_rules(self, img, margin, line_h, xh):
        d = ImageDraw.Draw(img)
        y = margin + int(1.42 * xh) + 3
        while y < img.height - margin:
            d.line([(margin - 20, y), (img.width - margin + 20, y)],
                   fill=(205, 214, 228), width=1)
            y += line_h

    # ---- public API --------------------------------------------------------
    def render(self, text, xh=26, page_w=1000, margin=70,
               ink=(20, 24, 60), bg=(252, 250, 244), ruled=False,
               slant=0.0):
        """Plain text -> single PIL image (grows to fit, no pagination)."""
        blocks = [{"type": "para", "runs": [("t", ln)] if ln else [("t", " ")],
                   "gap": 0.0} for ln in text.split("\n")]
        pages = self.render_document(blocks, xh, page_w, margin, ink, bg, ruled, slant)
        return self._merge_tall(pages, page_w, bg) if len(pages) > 1 else pages[0]

    def _merge_tall(self, pages, page_w, bg):
        h = sum(p.height for p in pages)
        out = Image.new("RGB", (page_w, h), bg)
        y = 0
        for p in pages:
            out.paste(p, (0, y)); y += p.height
        return out


def render_text(text, out_path, **kw):
    r = HandwritingRenderer(seed=kw.pop("seed", None), model=kw.pop("model", None))
    img = r.render(text, **kw)
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys
    txt = sys.argv[1] if len(sys.argv) > 1 else (
        "The quick brown fox jumps over the lazy dog.\n"
        "Economia e mercados financeiros: taxa de juros e inflacao.")
    out = os.path.join(os.path.dirname(HERE), "out", "sample_render.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render_text(txt, out, ruled=True, seed=7)
    print("wrote", out)
