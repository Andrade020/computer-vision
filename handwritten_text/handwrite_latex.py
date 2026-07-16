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
    ap.add_argument("--math-style", type=float, default=1.0,
                    help="symbol hand-styling strength: 0=clean, 1=default, 1.5+=rougher")
    ap.add_argument("--regularize", type=float, default=1.0,
                    help="even out glyph stroke weight: 0=raw, 1=normalized")
    ap.add_argument("--stroke", type=float, default=0.11,
                    help="target stroke width as fraction of x-height")
    ap.add_argument("--ink", type=float, default=0.8,
                    help="pen ink density texture within each stroke: 0=flat, 1=full")
    ap.add_argument("--tremor", type=float, default=0.3,
                    help="extra shape wobble on real glyphs: 0=off (raw bank shape)")
    ap.add_argument("--scan", type=float, nargs="?", const=1.0, default=0.0,
                    help="paper-scan look (warp/creases/grain/lighting drift): "
                         "0=off (default), bare flag=1.0, or give a strength")
    ap.add_argument("--title", default=None,
                    help="document title, stamped in the top margin of every page "
                         "in the same handwriting as the body text")
    ap.add_argument("--no-page-numbers", action="store_true",
                    help="disable the page-number footer (numbered from 1 by default)")
    args = ap.parse_args()

    blocks = parse_file(args.tex)
    print(f"parsed {len(blocks)} blocks from {args.tex}")

    model = None
    if args.model and os.path.exists("hw/checkpoints/best.pt"):
        from hw.model import GlyphGenerator
        model = GlyphGenerator("hw/checkpoints/best.pt")

    MARGIN = 70
    r = HandwritingRenderer(seed=args.seed, model=model, hand_math=args.hand_math,
                            math_style=args.math_style, regularize=args.regularize,
                            stroke_ratio=args.stroke, ink_texture=args.ink,
                            letter_tremor=args.tremor)
    pages = r.render_document(blocks, xh=args.xh, page_w=args.width, margin=MARGIN,
                              ruled=args.ruled, slant=args.slant)

    if args.title or not args.no_page_numbers:
        # stamped BEFORE the paper-scan warp below, so the title/page number
        # gets distorted along with the rest of the page instead of looking
        # like a crisp overlay pasted onto a warped scan
        for i, p in enumerate(pages, 1):
            r.stamp_header_footer(p, margin=MARGIN, xh=args.xh, title=args.title,
                                  page_num=None if args.no_page_numbers else i)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.scan > 0:
        from hw.paper import scan_effect
        base_seed = args.seed if args.seed is not None else 0
        pages = [scan_effect(p, strength=args.scan, seed=base_seed + i)
                for i, p in enumerate(pages)]
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
