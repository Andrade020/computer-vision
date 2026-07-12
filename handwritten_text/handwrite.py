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

    r = HandwritingRenderer(seed=args.seed, model=model,
                            regularize=args.regularize, stroke_ratio=args.stroke)
    img = r.render(text, xh=args.xh, page_w=args.width, ruled=args.ruled,
                   slant=args.slant)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    img.save(args.out)
    print("wrote", args.out, img.size)


if __name__ == "__main__":
    main()
