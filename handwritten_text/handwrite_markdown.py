"""
CLI: render a Markdown+LaTeX document (like the Elcyon econometrics resolution
lists) into handwriting -- one PNG per page, streamed to disk as each page
completes, plus a combined PDF at the end.

Built lazy/streaming and resilient on purpose: these source documents run
hundreds of math expressions across dozens of pages. A single-shot in-memory
render of that much LaTeX used to look "stuck" with no feedback and lost all
progress if anything went wrong partway through. Now:
  - every finished page is saved to disk the moment it's ready
  - progress prints per top-level block, so a long run stays visibly alive
  - a single malformed construct is logged and skipped, not fatal

  python handwrite_markdown.py doc.md -o out/doc --ruled --hand-math
"""
import argparse
import glob
import os
import re
import time

from PIL import Image

from hw.markdown_render import parse_file
from hw.render import HandwritingRenderer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("-o", "--out", default="out/doc", help="output basename")
    ap.add_argument("--xh", type=int, default=24)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--ruled", action="store_true")
    ap.add_argument("--slant", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--hand-math", action="store_true",
                    help="render math in your handwriting, not just typeset")
    ap.add_argument("--math-style", type=float, default=1.0)
    ap.add_argument("--regularize", type=float, default=1.0)
    ap.add_argument("--stroke", type=float, default=0.11)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop after N pages (useful for a quick preview)")
    ap.add_argument("--scan", type=float, nargs="?", const=1.0, default=0.0,
                    help="paper-scan look (warp/creases/grain/lighting drift): "
                         "0=off (default), bare flag=1.0, or give a strength")
    args = ap.parse_args()

    blocks = parse_file(args.md)
    print(f"parsed {len(blocks)} blocks from {args.md}", flush=True)

    model = None
    if args.model and os.path.exists("hw/checkpoints/best.pt"):
        from hw.model import GlyphGenerator
        model = GlyphGenerator("hw/checkpoints/best.pt")

    r = HandwritingRenderer(seed=args.seed, model=model, hand_math=args.hand_math,
                            math_style=args.math_style, regularize=args.regularize,
                            stroke_ratio=args.stroke)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    errors = []
    total = len(blocks)

    def on_block(i, n, blk):
        if i % args.progress_every == 0 or i == n - 1:
            print(f"  [{i + 1}/{n}] {blk.get('type')}", flush=True)

    def on_error(i, blk, exc):
        errors.append((i, blk.get("type"), str(exc)))
        print(f"  !! block {i} ({blk.get('type')}) failed, skipped: {exc}",
             flush=True)

    t0 = time.time()
    paths = []
    gen = r.iter_document(blocks, xh=args.xh, page_w=args.width, ruled=args.ruled,
                          slant=args.slant, on_block=on_block, on_error=on_error)
    scan_fn = None
    if args.scan > 0:
        from hw.paper import scan_effect
        base_seed = args.seed if args.seed is not None else 0
        scan_fn = lambda img, i: scan_effect(img, strength=args.scan, seed=base_seed + i)

    for pi, page in enumerate(gen, 1):
        if scan_fn:
            page = scan_fn(page, pi)
        pp = f"{args.out}_p{pi}.png"
        page.save(pp)
        paths.append(pp)
        print(f"wrote {pp} ({page.size[0]}x{page.size[1]}) at {time.time() - t0:.1f}s",
             flush=True)
        if args.max_pages and pi >= args.max_pages:
            print(f"stopping early: --max-pages={args.max_pages}")
            break

    print(f"rendered {len(paths)} page(s) in {time.time() - t0:.1f}s, "
         f"{len(errors)} block(s) skipped", flush=True)

    if paths:
        pages = [Image.open(p).convert("RGB") for p in paths]
        pdf_path = f"{args.out}.pdf"
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:])
        print(f"wrote {pdf_path} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
