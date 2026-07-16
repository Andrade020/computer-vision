"""
CLI: render plain text as handwriting.

  python handwrite.py "your text here" -o out/hello.png
  python handwrite.py -f notes.txt -o out/notes.png --ruled --model
"""
import argparse
import os
from hw.render import HandwritingRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default=None)
    ap.add_argument("-f", "--file")
    ap.add_argument("-o", "--out", default="out/handwriting.png")
    ap.add_argument("--xh", type=int, default=26, help="x-height in px")
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--ruled", action="store_true")
    ap.add_argument("--slant", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", action="store_true", help="use neural fallback for missing glyphs")
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
                    help="document title, stamped in the top margin in the same "
                         "handwriting as the body text")
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as fp:
            text = fp.read()
    elif args.text:
        text = args.text
    else:
        text = "Hello! This is my handwriting, synthesized by a neural pipeline."

    model = None
    if args.model:
        from hw.model import GlyphGenerator
        ck = "hw/checkpoints/best.pt"
        if os.path.exists(ck):
            model = GlyphGenerator(ck)
        else:
            print("(no checkpoint yet; using real-ink bank only)")

    MARGIN = 70
    r = HandwritingRenderer(seed=args.seed, model=model,
                            regularize=args.regularize, stroke_ratio=args.stroke,
                            ink_texture=args.ink, letter_tremor=args.tremor)
    img = r.render(text, xh=args.xh, page_w=args.width, margin=MARGIN,
                   ruled=args.ruled, slant=args.slant)
    if args.title:
        r.stamp_header_footer(img, margin=MARGIN, xh=args.xh, title=args.title)
    if args.scan > 0:
        from hw.paper import scan_effect
        img = scan_effect(img, strength=args.scan, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print("wrote", args.out, img.size)


if __name__ == "__main__":
    main()
