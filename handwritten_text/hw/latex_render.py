"""
LaTeX -> handwriting document.

Parses a curated but common subset of LaTeX and produces "blocks" for
HandwritingRenderer.render_document: prose is handwritten in the user's
glyphs, math ($...$, \\[...\\], equation/align) is typeset via mathtext and
spliced inline / centered. Robust to unknown commands (they are stripped).

This is NOT a full TeX engine -- it targets notes/quizzes/homework-style docs.
"""
import re
import os
import unicodedata

MATH_ENVS = ["equation", "equation*", "align", "align*", "displaymath", "gather", "gather*"]


def _strip_accents(s):
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _remove_comments(tex):
    # drop % ... to EOL, but keep \%
    return re.sub(r"(?<!\\)%.*", "", tex)


def _body(tex):
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", tex, re.S)
    return m.group(1) if m else tex


def _protect_math(tex):
    """Replace math with placeholders; return (text, [(expr, display)])."""
    math = []

    def repl(expr, display):
        math.append((expr.strip(), display))
        return f"\x00M{len(math)-1}\x00"

    # display environments
    for env in MATH_ENVS:
        tex = re.sub(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}",
                     lambda m: repl(m.group(1), True), tex, flags=re.S)
    tex = re.sub(r"\\\[(.*?)\\\]", lambda m: repl(m.group(1), True), tex, flags=re.S)
    tex = re.sub(r"\$\$(.*?)\$\$", lambda m: repl(m.group(1), True), tex, flags=re.S)
    tex = re.sub(r"\\\((.*?)\\\)", lambda m: repl(m.group(1), False), tex, flags=re.S)
    tex = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", lambda m: repl(m.group(1), False), tex, flags=re.S)
    return tex, math


def _clean_text(s):
    # strip formatting commands but keep their argument
    for cmd in ["textbf", "textit", "emph", "underline", "text", "mathrm",
                "texttt", "textsc", "mbox", "textrm"]:
        s = re.sub(r"\\" + cmd + r"\{([^{}]*)\}", r"\1", s)
    # common escapes
    repl = {r"\%": "%", r"\&": "&", r"\_": "_", r"\#": "#", r"\$": "$",
            r"\{": "{", r"\}": "}", r"\ldots": "...", r"\dots": "...",
            r"\textbackslash": "/", r"~": " ", r"``": '"', r"''": '"'}
    for a, b in repl.items():
        s = s.replace(a, b)
    # accented control sequences like \'a \~a \c{c}
    s = re.sub(r"\\[`'^\"~=.]\{?([a-zA-Z])\}?", r"\1", s)
    s = re.sub(r"\\c\{([a-zA-Z])\}", r"\1", s)
    # drop any remaining unknown commands (with or without args)
    s = re.sub(r"\\[a-zA-Z]+\*?(\{[^{}]*\})?", " ", s)
    s = _strip_accents(s)
    s = s.replace("--", "-")
    return s


def _runs_from_text(text, math):
    """Split a text chunk (with math placeholders) into runs."""
    runs = []
    parts = re.split(r"\x00M(\d+)\x00", text)
    for i, part in enumerate(parts):
        if i % 2 == 1:                       # math index
            expr, display = math[int(part)]
            runs.append(("m", expr, display))
        else:
            # line breaks
            for j, seg in enumerate(re.split(r"\\\\|\\newline", part)):
                if j > 0:
                    runs.append(("br",))
                seg = _clean_text(seg)
                seg = re.sub(r"[ \t]+", " ", seg).strip()
                if seg:
                    runs.append(("t", seg))
    return [r for r in runs if r != ("t", "")]


def parse(tex):
    """Return a list of blocks for render_document."""
    tex = _remove_comments(tex)
    tex = _body(tex)
    tex, math = _protect_math(tex)

    blocks = []

    # pull out title
    mt = re.search(r"\\title\{([^{}]*)\}", tex)
    if mt:
        blocks.append({"type": "heading", "scale": 1.6, "gap": 0.4,
                       "runs": _runs_from_text(mt.group(1), math)})
        tex = tex.replace(mt.group(0), "")
    tex = re.sub(r"\\maketitle|\\author\{[^{}]*\}|\\date\{[^{}]*\}", "", tex)

    # walk the body, splitting on section headers and list environments
    # tokenize into a linear stream
    pattern = re.compile(
        r"(\\section\*?\{[^{}]*\}|\\subsection\*?\{[^{}]*\}|"
        r"\\begin\{itemize\}|\\end\{itemize\}|"
        r"\\begin\{enumerate\}|\\end\{enumerate\}|\\item)")
    tokens = pattern.split(tex)

    list_depth = 0
    buf = ""

    def flush_para(text):
        for chunk in re.split(r"\n\s*\n", text):
            runs = _runs_from_text(chunk, math)
            if runs:
                # separate display math into their own centered blocks
                _emit_runs(blocks, runs, indent=list_depth)

    for tok in tokens:
        if tok is None:
            continue
        if tok.startswith("\\section"):
            flush_para(buf); buf = ""
            title = re.search(r"\{([^{}]*)\}", tok)
            blocks.append({"type": "heading", "scale": 1.45, "gap": 0.35,
                           "newline_before": True, "level": 1,
                           "runs": _runs_from_text(title.group(1) if title else "", math)})
        elif tok.startswith("\\subsection"):
            flush_para(buf); buf = ""
            title = re.search(r"\{([^{}]*)\}", tok)
            blocks.append({"type": "heading", "scale": 1.2, "gap": 0.3, "level": 2,
                           "runs": _runs_from_text(title.group(1) if title else "", math)})
        elif tok.startswith("\\begin{itemize}") or tok.startswith("\\begin{enumerate}"):
            flush_para(buf); buf = ""
            list_depth += 1
        elif tok.startswith("\\end{itemize}") or tok.startswith("\\end{enumerate}"):
            flush_para(buf); buf = ""
            list_depth = max(0, list_depth - 1)
        elif tok == "\\item":
            flush_para(buf); buf = ""
            buf = "\x01ITEM\x01"
        else:
            buf += tok
    flush_para(buf)
    return blocks


def _emit_runs(blocks, runs, indent=0):
    """Emit runs, breaking display math into centered blocks; bullets for items."""
    bullet = False
    if runs and runs[0] == ("t", "\x01ITEM\x01".strip()):
        pass
    # detect item marker embedded as text
    cleaned = []
    for r in runs:
        if r[0] == "t" and "\x01ITEM\x01" in r[1]:
            bullet = True
            txt = r[1].replace("\x01ITEM\x01", "").strip()
            if txt:
                cleaned.append(("t", txt))
        else:
            cleaned.append(r)

    # split leading/trailing display math into own blocks
    line_runs = []
    for r in cleaned:
        if r[0] == "m" and r[2]:            # display math
            if line_runs:
                blocks.append(_mk(line_runs, indent, bullet)); bullet = False
                line_runs = []
            blocks.append({"type": "displaymath", "runs": [r],
                           "center": True, "gap": 0.3, "scale": 1.0})
        else:
            line_runs.append(r)
    if line_runs:
        blocks.append(_mk(line_runs, indent, bullet))


def _mk(runs, indent, bullet):
    return {"type": "para", "runs": runs, "indent": indent,
            "bullet": bullet, "gap": 0.15}


def parse_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return parse(f.read())
