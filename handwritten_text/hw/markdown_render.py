"""
Parse the Pandoc-style Markdown used for these resolution documents (headers
#/##/###, **bold**, *italic*, --- rules, numbered/bulleted lists, and
$...$ / $$...$$ math) into the block list HandwritingRenderer.render_document
consumes.

Deliberately tolerant: unrecognized lines fall back to plain paragraph text
rather than raising, so one odd line never derails a long document. Math
extraction/cleanup mirrors hw/latex_render.py; this module differs mainly in
recognizing Markdown block syntax instead of LaTeX \\section/\\begin{itemize}.
"""
import os
import re
import unicodedata

_HEADING_SCALE = {1: 1.6, 2: 1.45, 3: 1.28, 4: 1.16, 5: 1.06, 6: 1.0}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^-{3,}\s*$")
_NUM_RE = re.compile(r"^(\d+)\.\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)-\s+(.*)$")
_BLANK_RE = re.compile(r"^\s*$")
_IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")
_TABLE_PLACEHOLDER_RE = re.compile(r"^\x00TABLE(\d+)\x00$")


def _split_table_row(line):
    """'| a | b |' -> ['a', 'b'] -- strip the outer pipes, then split/trim
    each cell. Cells are treated as plain text (no bold/italic/math markup
    resolution inside table cells in this first version -- see the
    "Limitações honestas" note in README_handwriting.md)."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [_normalize_text(_strip_markup(c)) for c in inner.split("|")]


def _extract_tables(lines):
    """Scan for contiguous GFM-style pipe tables (a header row, a
    ``|---|---|`` separator, then 1+ data rows) and pull each one out into
    ``tables``, replacing its lines in the stream with a single placeholder
    line the main parse loop recognizes. Tables are multi-line constructs
    that need lookahead (the separator row on the line *after* the header
    is what confirms "this is a table, not a paragraph starting with a
    pipe character"), which doesn't fit the single-pass, no-lookahead state
    machine the rest of parse() uses -- pulling them out first keeps that
    loop unchanged."""
    out_lines = []
    tables = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if (_TABLE_ROW_RE.match(line) and i + 1 < n
                and _TABLE_SEP_RE.match(lines[i + 1].strip())):
            header = _split_table_row(line)
            j = i + 2
            rows = [header]
            while j < n and _TABLE_ROW_RE.match(lines[j].strip()):
                rows.append(_split_table_row(lines[j].strip()))
                j += 1
            tables.append(rows)
            out_lines.append(f"\x00TABLE{len(tables) - 1}\x00")
            i = j
        else:
            out_lines.append(lines[i])
            i += 1
    return out_lines, tables


def _strip_accents(s):
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _protect_math(text):
    """Replace $...$ / $$...$$ spans with placeholders; return (text, [(expr, display)])."""
    math = []

    def repl(expr, display):
        math.append((expr.strip(), display))
        return f"\x00M{len(math) - 1}\x00"

    text = re.sub(r"\$\$(.+?)\$\$", lambda m: repl(m.group(1), True), text, flags=re.S)
    text = re.sub(r"(?<!\$)\$(.+?)(?<!\$)\$", lambda m: repl(m.group(1), False),
                 text, flags=re.S)
    return text, math


def _strip_markup(s):
    """Strip markdown emphasis/bold/code markers and normalize dashes/quotes.
    Runs on the FULL text (placeholders and all) so a bold span that straddles
    an inline-math placeholder -- e.g. "**mean of $Y$:**" -- still matches as
    one pair instead of leaving stray ** behind on each side."""
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)              # **bold**
    s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", r"\1", s)  # *italic*
    s = re.sub(r"__(.+?)__", r"\1", s)
    s = re.sub(r"(?<!_)_([^_\n]+?)_(?!_)", r"\1", s)
    s = re.sub(r"`([^`]+?)`", r"\1", s)                  # `code`
    s = s.replace("—", " - ").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")
    return s


def _normalize_text(s):
    s = _strip_accents(s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def _runs_from_text(text, math):
    """Split a placeholder-bearing chunk into ('t',...)/('m',expr,disp)/('br',) runs."""
    text = _strip_markup(text)
    runs = []
    for si, seg in enumerate(text.split("\x00BR\x00")):
        if si > 0:
            runs.append(("br",))
        parts = re.split(r"\x00M(\d+)\x00", seg)
        for i, part in enumerate(parts):
            if i % 2 == 1:
                expr, display = math[int(part)]
                runs.append(("m", expr, display))
            else:
                cleaned = _normalize_text(part)
                if cleaned:
                    runs.append(("t", cleaned))
    return runs


class _State:
    """Accumulates the lines of the paragraph/list-item currently being read."""
    def __init__(self):
        self.lines = []          # list of (text, hard_break_after)
        self.indent = 0
        self.bullet = False
        self.bullet_text = None  # e.g. "3." for an ordered list item

    def add(self, text, hard_break):
        self.lines.append((text, hard_break))

    def is_empty(self):
        return not self.lines

    def raw(self):
        out = ""
        for i, (txt, hard) in enumerate(self.lines):
            if i > 0:
                out += "\x00BR\x00" if self.lines[i - 1][1] else " "
            out += txt
        return out


def parse(text, base_dir=None):
    """Return a list of blocks for HandwritingRenderer.render_document.

    ``base_dir``, if given, resolves relative image paths in
    ``![caption](path)`` syntax against that directory (``parse_file``
    passes the source .md file's own directory, so
    ``![x](figs/plot.png)`` works relative to the document, not the
    current working directory the CLI happens to be run from).
    """
    raw_lines, tables = _extract_tables(text.split("\n"))
    text = "\n".join(raw_lines)
    text, math = _protect_math(text)
    lines = text.split("\n")

    blocks = []
    st = _State()

    def flush():
        nonlocal st
        if not st.is_empty():
            runs = _runs_from_text(st.raw(), math)
            if runs:
                blk = {"type": "para", "runs": runs, "indent": st.indent,
                      "gap": 0.15}
                if st.bullet_text:
                    blk["bullet_text"] = st.bullet_text
                elif st.bullet:
                    blk["bullet"] = True
                elif len(runs) == 1 and runs[0][0] == "m" and runs[0][2]:
                    blk["center"] = True    # a lone display-math line reads as an equation
                blocks.append(blk)
        st = _State()

    for raw_line in lines:
        hard_break = raw_line.endswith("  ") and raw_line.strip() != ""
        line = raw_line.rstrip()

        if _BLANK_RE.match(line):
            flush()
            continue

        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            runs = _runs_from_text(m.group(2), math)
            if runs:
                blocks.append({"type": "heading", "scale": _HEADING_SCALE[level],
                              "gap": 0.35 if level <= 2 else 0.22,
                              "newline_before": level <= 2, "runs": runs,
                              "level": level})
            continue

        if _HR_RE.match(line):
            flush()
            blocks.append({"type": "rule", "gap": 0.25})
            continue

        m = _TABLE_PLACEHOLDER_RE.match(line)
        if m:
            flush()
            blocks.append({"type": "table", "rows": tables[int(m.group(1))],
                          "header": True, "gap": 0.3})
            continue

        m = _IMAGE_RE.match(line)
        if m:
            flush()
            caption, path = m.group(1).strip(), m.group(2).strip()
            if base_dir and not os.path.isabs(path):
                path = os.path.join(base_dir, path)
            blocks.append({"type": "figure", "path": path,
                          "caption": caption or None, "gap": 0.3})
            continue

        m = _NUM_RE.match(line)
        if m:
            flush()
            st.bullet_text = m.group(1) + "."
            st.add(m.group(2), hard_break)
            continue

        m = _BULLET_RE.match(line)
        if m:
            flush()
            st.indent = 1 if len(m.group(1)) >= 2 else 0
            st.bullet = True
            st.add(m.group(2), hard_break)
            continue

        # continuation of the paragraph / list item currently being read
        st.add(line.strip(), hard_break)

    flush()
    return blocks


def parse_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read(), base_dir=os.path.dirname(os.path.abspath(path)))
