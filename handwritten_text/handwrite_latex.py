"""
CLI: compile a LaTeX document into handwriting.

Prose is rendered in your handwriting; math ($...$, \\[...\\], equation/align)
is typeset and spliced in. Outputs one PNG per page and a combined PDF.

  python handwrite_latex.py doc.tex -o out/doc
  python handwrite_latex.py doc.tex -o out/doc --ruled --model
"""
import argparse
import os
from hw.latex_render import parse_file
from hw.render import HandwritingRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tex")
    ap.add_argument("-o", "--out", default="out/doc", help="output basename")
    ap.add_argument("--xh", type=int, default=24)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--ruled", action="store_true")
    ap.add_argument("--slant", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--hand-math", action="store_true",
                    help="render math in your handwriting (real glyphs + inked symbols)")
    args = ap.parse_args()

    blocks = parse_file(args.tex)
    print(f"parsed {len(blocks)} blocks from {args.tex}")

    model = None
    if args.model and os.path.exists("hw/checkpoints/best.pt"):
        from hw.model import GlyphGenerator
        model = GlyphGenerator("hw/checkpoints/best.pt")

    r = HandwritingRenderer(seed=args.seed, model=model, hand_math=args.hand_math)
    pages = r.render_document(blocks, xh=args.xh, page_w=args.width,
                              ruled=args.ruled, slant=args.slant)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    paths = []
    for i, p in enumerate(pages, 1):
        pp = f"{args.out}_p{i}.png"
        p.save(pp)
        paths.append(pp)
        print("wrote", pp, p.size)
    if pages:
        pdf = f"{args.out}.pdf"
        pages[0].save(pdf, save_all=True, append_images=pages[1:])
        print("wrote", pdf, f"({len(pages)} pages)")


if __name__ == "__main__":
    main()
