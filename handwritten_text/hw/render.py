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
                 math_style=1.0, regularize=1.0, stroke_ratio=0.11, ink_texture=0.8,
                 letter_tremor=0.3):
        with open(bank_path, "rb") as f:
            d = pickle.load(f)
        self.bank = d["bank"]
        self.classes = set(d["classes"])
        self.model = model
        self.hand_math = hand_math          # render math in the user's hand
        self.math_style = math_style        # 0=clean symbols .. 1=default .. more=rougher
        self.regularize = regularize        # 0=raw glyph weight .. 1=fully normalized
        self.stroke_ratio = stroke_ratio    # target stroke width as fraction of x-height
        self.ink_texture = ink_texture      # 0=flat ink .. 1=full pen-like density variation
        self.letter_tremor = letter_tremor  # 0=off .. extra shape wobble on real glyphs too
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
        """Resize a glyph to its render box, add a touch of shape wobble,
        normalize its stroke weight so all glyphs share a consistent
        thickness, and give the ink a pen-like density texture. Returns
        (mask, w, h, dy) where dy is the vertical growth to fold into the
        baseline offset."""
        g = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                       .resize((w_px, h_px), Image.LANCZOS), np.float32) / 255.0
        dy = 0.0
        if self.letter_tremor > 0 or self.regularize > 0:
            from .imageops import normalize_stroke, elastic
            pad = 6
            g = np.pad(g, pad)
            if self.letter_tremor > 0:
                g = elastic(g, sigma=0.5 * xh, amp=0.018 * self.letter_tremor * xh,
                           rng=self._npr)
            if self.regularize > 0:
                g = normalize_stroke(g, self.stroke_ratio * xh, self.regularize)
            ys, xs = np.where(g > 0.12)
            if len(xs):
                g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                h2, w2 = g.shape
                w_px, h_px, dy = w2, h2, (h2 - h_px) / 2.0
            else:
                g = g[pad:pad + h_px, pad:pad + w_px]
        if self.ink_texture > 0:
            from .imageops import ink_texture
            g = ink_texture(g, strength=self.ink_texture, rng=self._npr)
        return g, w_px, h_px, dy

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
                      slant=0.0, on_block=None, on_error=None, on_heading=None):
        """Yield finished PIL pages one at a time as they fill, instead of
        holding the whole document in memory. This is the "lazy" entry point:
        callers (see the CLIs) save each page to disk the moment it arrives,
        so a long document that fails partway through still leaves every
        completed page on disk instead of losing everything.

        Each block is wrapped in try/except: a single malformed construct
        (reported via on_error(index, block, exc) if given) just ends the
        current line and moves on, instead of aborting the whole document.
        on_block(index, total, block), if given, fires before each block --
        use it to print progress on a long run. on_heading(text, level,
        page_num), if given, fires for every heading block that carries a
        "level" key (markdown_render.py/latex_render.py both set this),
        reporting which 1-based page it landed on -- this is how
        render_document_with_toc (below) builds a table of contents without
        needing a second, separate layout pass just to find page numbers.
        """
        line_h = int((1.42 + 0.42 + 1.0) * xh)
        base_desc = int(0.42 * xh)          # descent a normal text line already reserves
        max_x = page_w - margin
        page_h = int(297.0 / 210.0 * page_w)       # A4 aspect (ISO 216)
        ready = []
        line_extra_desc = 0    # extra room a tall inline image (e.g. a matrix) needs

        def new_page():
            img = Image.new("RGB", (page_w, page_h), bg)
            if ruled:
                self._draw_rules(img, margin, line_h, xh)
            return img

        img = new_page()
        y = margin + int(1.42 * xh)
        x = margin
        page_num = 1   # 1-based; incremented every time a new physical page starts

        def advance_line(cur_y):
            nonlocal img, line_extra_desc, page_num
            ny = cur_y + line_h + line_extra_desc
            line_extra_desc = 0
            if ny > page_h - margin:
                ready.append(img)
                img = new_page()
                page_num += 1
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
                elif kind == "figure":
                    if x != margin:
                        y = advance_line(y)
                    fig_img = Image.open(blk["path"]).convert("RGBA")
                    avail_w = max_x - margin
                    max_h = int(0.55 * page_h)
                    scale = min(avail_w / fig_img.width, max_h / fig_img.height, 1.0)
                    new_w = max(1, int(fig_img.width * scale))
                    new_h = max(1, int(fig_img.height * scale))
                    if scale != 1.0:
                        fig_img = fig_img.resize((new_w, new_h), Image.LANCZOS)
                    if y + new_h > page_h - margin and y > margin + int(1.42 * xh):
                        ready.append(img)
                        img = new_page()
                        page_num += 1
                        y = margin + int(1.42 * xh)
                    fx = margin + (avail_w - new_w) // 2
                    img.paste(fig_img, (int(fx), int(y)), fig_img)
                    y += new_h + int(0.15 * xh)
                    caption = blk.get("caption")
                    if caption:
                        cap_xh = max(6, int(xh * 0.82))
                        cap_units = self._text_to_units(caption, cap_xh)
                        cap_w = sum(u["w"] for u in cap_units)
                        cx = margin + max(0, (avail_w - cap_w) // 2)
                        cap_baseline = y + int(1.0 * cap_xh)
                        self._draw_units_line(img, cap_units, cx, cap_baseline, cap_xh,
                                              ink, slant, max_x=max_x)
                        y = cap_baseline + int(0.42 * cap_xh)
                    y = min(page_h - margin, y + int(blk.get("gap", 0.3) * line_h))
                    x = margin
                elif kind == "table":
                    rows = blk["rows"]
                    ncols = max(len(r) for r in rows) if rows else 0
                    if ncols == 0:
                        continue
                    if x != margin:
                        y = advance_line(y)
                    avail_w = max_x - margin
                    col_w = avail_w // ncols
                    cell_xh = max(6, int(xh * 0.85))
                    row_h = int(1.7 * cell_xh)
                    table_h = row_h * len(rows)
                    if y + table_h > page_h - margin and y > margin + int(1.42 * xh):
                        ready.append(img)
                        img = new_page()
                        page_num += 1
                        y = margin + int(1.42 * xh)
                    top_y = y
                    is_header_row = blk.get("header", False)
                    for ri, row in enumerate(rows):
                        ry = top_y + ri * row_h
                        for ci in range(ncols):
                            cell_text = row[ci] if ci < len(row) else ""
                            if not cell_text:
                                continue
                            cx0 = margin + ci * col_w + int(0.15 * cell_xh)
                            cell_units = self._text_to_units(cell_text, cell_xh)
                            baseline = ry + int(1.1 * cell_xh)
                            self._draw_units_line(
                                img, cell_units, cx0, baseline, cell_xh, ink, slant,
                                max_x=margin + (ci + 1) * col_w - int(0.1 * cell_xh))
                    grid = ImageDraw.Draw(img)
                    for ri in range(len(rows) + 1):
                        ry = top_y + ri * row_h
                        w = 2 if (is_header_row and ri <= 1) else 1
                        grid.line([(margin, ry), (margin + ncols * col_w, ry)],
                                 fill=(190, 196, 212), width=w)
                    for ci in range(ncols + 1):
                        cxx = margin + ci * col_w
                        grid.line([(cxx, top_y), (cxx, top_y + len(rows) * row_h)],
                                 fill=(190, 196, 212), width=1)
                    y = top_y + len(rows) * row_h
                    y = min(page_h - margin, y + int(blk.get("gap", 0.3) * line_h))
                    x = margin
                else:
                    bxh = int(xh * blk.get("scale", 1.0))
                    indent = blk.get("indent", 0) * int(1.4 * xh)
                    x = margin + indent
                    if blk.get("newline_before") and x != margin:
                        y = advance_line(y)

                    if kind == "heading" and "level" in blk and on_heading:
                        heading_text = " ".join(r[1] for r in blk["runs"] if r[0] == "t")
                        if heading_text:
                            on_heading(heading_text, blk["level"], page_num)

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
                            desc_px = u["h"] - u["asc"]
                            if desc_px > base_desc:
                                line_extra_desc = max(line_extra_desc, desc_px - base_desc)
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
                        slant=0.0, on_heading=None, on_block=None, on_error=None):
        """Rich blocks -> list of PIL pages (A4 aspect). For long documents
        prefer iter_document directly so pages can be saved as they're produced."""
        return list(self.iter_document(blocks, xh, page_w, margin, ink, bg, ruled, slant,
                                       on_block=on_block, on_error=on_error,
                                       on_heading=on_heading))

    def build_toc_blocks(self, entries, title="Sumario"):
        """Turn a list of (text, level, page_num) heading events -- gathered
        via iter_document's on_heading callback -- into blocks for a table-
        of-contents page, formatted as "Heading text .... 3" with a dot
        leader and indentation by level. Each entry is a single-line
        paragraph, so the same word-wrap-avoidance dot-count heuristic below
        is deliberately approximate (hand-written glyph widths vary, so
        counting characters can't line up page numbers pixel-perfectly the
        way a real typeset TOC with tab stops would) -- good enough to read
        by eye, not typographically exact.
        """
        blocks = [{"type": "heading", "scale": 1.6, "gap": 0.4, "runs": [("t", title)]}]
        target_chars = 62
        for text, level, page_num in entries:
            indent = max(0, min(level - 1, 3))
            dots = max(3, target_chars - len(text) - len(str(page_num)) - indent * 3)
            line = f"{text} {'.' * dots} {page_num}"
            blocks.append({"type": "para", "runs": [("t", line)], "indent": indent,
                          "gap": 0.12})
        return blocks

    def render_document_with_toc(self, blocks, toc_title="Sumario", toc_levels=(1, 2),
                                 xh=26, page_w=1000, margin=70, ink=(20, 24, 60),
                                 bg=(252, 250, 244), ruled=False, slant=0.0,
                                 on_block=None, on_error=None):
        """Two-pass document assembly: render the content once (buffered, not
        lazy -- building a TOC inherently needs to know page numbers for
        headings that only become known by actually laying out the whole
        document first, so this trades away iter_document's streaming
        property on purpose), collecting (heading text, level, page number)
        along the way, then renders a second, small document for the table
        of contents from those collected entries.

        Returns (toc_pages, content_pages) as two separate lists, rather
        than one combined list -- deliberately, so callers can number them
        independently: content pages as "Pagina 1, 2, 3..." and TOC pages
        left unnumbered (or numbered separately, e.g. roman numerals),
        matching how a lot of real documents treat front matter. Otherwise,
        content page numbers would have to shift by however many pages the
        TOC itself ends up taking -- which isn't known until the TOC is
        rendered, i.e. a bootstrapping problem this side-steps entirely.
        """
        toc_entries = []

        def on_heading(text, level, page_num):
            if level in toc_levels:
                toc_entries.append((text, level, page_num))

        content_pages = self.render_document(blocks, xh=xh, page_w=page_w, margin=margin,
                                             ink=ink, bg=bg, ruled=ruled, slant=slant,
                                             on_heading=on_heading, on_block=on_block,
                                             on_error=on_error)
        toc_blocks = self.build_toc_blocks(toc_entries, title=toc_title)
        toc_pages = self.render_document(toc_blocks, xh=xh, page_w=page_w, margin=margin,
                                         ink=ink, bg=bg, ruled=ruled, slant=slant)
        return toc_pages, content_pages

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

    def _draw_units_line(self, img, units, x0, baseline_y, xh, ink, slant, max_x=None):
        """Draw a single line of layout units (glyphs/inline images/spaces)
        starting at x0 along one baseline, with no wrapping -- lines that
        run past ``max_x`` are simply truncated instead of continuing onto
        a new line. This is deliberately simpler than the main flow in
        iter_document (which wraps across lines): it exists for short,
        one-off stamps -- page headers/footers, table cells, TOC entries --
        that are always meant to fit on a single line."""
        x = x0
        for u in units:
            if u["kind"] == "space":
                x += u["w"]
                continue
            if max_x is not None and x + u["w"] > max_x:
                break
            if u["kind"] == "glyphs":
                self._paste_word(img, u, x, baseline_y, xh, ink, slant)
            elif u["kind"] == "image":
                top = baseline_y - u["asc"]
                img.paste(u["img"], (int(x), int(top)), u["img"])
            x += u["w"]
        return x

    def stamp_header_footer(self, img, margin, xh, ink=(20, 24, 60), title=None,
                            page_num=None, slant=0.0):
        """Draw a small document title (top margin band) and/or page number
        (bottom margin band) directly onto an already-finished page, reusing
        the same hand-glyph pipeline as the body text -- so page furniture
        looks like part of the same handwritten document instead of a
        robotically stamped generic font. Operates in place on ``img``, and
        is meant to be called per page, after ``iter_document`` yields it
        (see handwrite.py/handwrite_latex.py/handwrite_markdown.py), which
        keeps the streaming/lazy architecture intact -- no changes needed to
        iter_document itself for this."""
        small_xh = max(8, int(xh * 0.6))
        if title:
            units = self._text_to_units(title, small_xh)
            baseline_y = max(int(small_xh * 1.1), int(margin * 0.62))
            self._draw_units_line(img, units, margin, baseline_y, small_xh, ink,
                                  slant, max_x=img.width - margin)
        if page_num is not None:
            units = self._text_to_units(str(page_num), small_xh)
            total_w = sum(u["w"] for u in units)
            x0 = max(margin, (img.width - total_w) // 2)
            baseline_y = img.height - max(int(small_xh * 0.6), int(margin * 0.35))
            self._draw_units_line(img, units, x0, baseline_y, small_xh, ink, slant)

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
