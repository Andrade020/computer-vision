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


class HandwritingRenderer:
    def __init__(self, bank_path=BANK_PATH, model=None, seed=None, hand_math=False):
        with open(bank_path, "rb") as f:
            d = pickle.load(f)
        self.bank = d["bank"]
        self.classes = set(d["classes"])
        self.model = model
        self.hand_math = hand_math          # render math in the user's hand
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
        return None

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
        top_px = int(round(top * xh))          # above baseline
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
    def _flow(self, blocks, xh, page_w, margin, ink, bg, ruled, slant, pages):
        """blocks: list produced by document builders. Returns list of PIL pages."""
        line_h = int((1.42 + 0.42 + 1.0) * xh)
        max_x = page_w - margin
        page_h = int(11.0 / 8.5 * page_w)          # letter aspect
        out_pages = []

        def new_page():
            img = Image.new("RGB", (page_w, page_h), bg)
            if ruled:
                self._draw_rules(img, margin, line_h, xh)
            return img

        img = new_page()
        y = margin + int(1.42 * xh)
        x = margin

        def wrap_if_needed(w):
            nonlocal x, y, img
            if x + w > max_x and x > margin:
                x = margin
                y = advance_line(y)

        def advance_line(cur_y):
            nonlocal img
            ny = cur_y + line_h
            if ny > page_h - margin:
                out_pages.append(img)
                img = new_page()
                return margin + int(1.42 * xh)
            return ny

        for blk in blocks:
            kind = blk["type"]
            if kind == "vspace":
                y = min(page_h - margin, y + blk["px"])
                x = margin
                continue
            bxh = int(xh * blk.get("scale", 1.0))
            indent = blk.get("indent", 0) * int(1.4 * xh)
            x = margin + indent
            if blk.get("newline_before") and x != margin:
                y = advance_line(y)

            units = self._blk_units(blk, bxh)
            if blk.get("bullet"):
                units = [self._dot_unit(bxh, ink), self._space_unit(bxh)] + units

            if blk.get("center"):
                total = sum(u["w"] for u in units)
                x = max(margin + indent, margin + (max_x - margin - total) // 2)

            for u in units:
                if u["kind"] == "break":
                    x = margin + indent
                    y = advance_line(y)
                    continue
                wrap_if_needed(u["w"])
                if u["kind"] == "space":
                    x += u["w"]
                    continue
                if u["kind"] == "glyphs":
                    self._paste_word(img, u, x, y, bxh, ink, slant)
                elif u["kind"] == "image":
                    top = y - u["asc"]
                    img.paste(u["img"], (int(x), int(top)), u["img"])
                x += u["w"]
            # end of block -> newline + a little gap
            y = advance_line(y)
            y = min(page_h - margin, y + int(blk.get("gap", 0.2) * line_h))
            x = margin

        out_pages.append(img)
        if pages:
            return out_pages
        return out_pages[:1] if len(out_pages) == 1 else out_pages

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
                                                   rng=self._npr, display=display)
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
        pages = self._flow(blocks, xh, page_w, margin, ink, bg, ruled, slant,
                            pages=True)
        return self._merge_tall(pages, page_w, bg) if len(pages) > 1 else pages[0]

    def render_document(self, blocks, xh=26, page_w=1000, margin=70,
                        ink=(20, 24, 60), bg=(252, 250, 244), ruled=False,
                        slant=0.0):
        """Rich blocks -> list of PIL pages (letter aspect)."""
        return self._flow(blocks, xh, page_w, margin, ink, bg, ruled, slant,
                          pages=True)

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
