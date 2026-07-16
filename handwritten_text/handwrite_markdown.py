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
    ap.add_argument("--ink", type=float, default=0.8,
                    help="pen ink density texture within each stroke: 0=flat, 1=full")
    ap.add_argument("--tremor", type=float, default=0.3,
                    help="extra shape wobble on real glyphs: 0=off (raw bank shape)")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop after N pages (useful for a quick preview)")
    ap.add_argument("--scan", type=float, nargs="?", const=1.0, default=0.0,
                    help="paper-scan look (warp/creases/grain/lighting drift): "
                         "0=off (default), bare flag=1.0, or give a strength")
    ap.add_argument("--title", default=None,
                    help="document title, stamped in the top margin of every page "
                         "in the same handwriting as the body text")
    ap.add_argument("--no-page-numbers", action="store_true",
                    help="disable the page-number footer (numbered from 1 by default)")
    ap.add_argument("--toc", nargs="?", const="Sumario", default=None, metavar="TITULO",
                    help="add a table-of-contents page before the content, tracking "
                         "#/## headings and the page each lands on; bare flag titles "
                         "it 'Sumario', or give a custom title. NOTE: building a TOC "
                         "needs to know page numbers ahead of time, so this switches "
                         "from the default lazy/streaming render (pages saved as they "
                         "complete, document never fully held in memory) to a buffered "
                         "one -- fine for most documents, but for a very long one where "
                         "memory is a concern, skip --toc and keep the default streaming "
                         "path")
    args = ap.parse_args()

    blocks = parse_file(args.md)
    print(f"parsed {len(blocks)} blocks from {args.md}", flush=True)

    model = None
    if args.model and os.path.exists("hw/checkpoints/best.pt"):
        from hw.model import GlyphGenerator
        model = GlyphGenerator("hw/checkpoints/best.pt")

    MARGIN = 70
    r = HandwritingRenderer(seed=args.seed, model=model, hand_math=args.hand_math,
                            math_style=args.math_style, regularize=args.regularize,
                            stroke_ratio=args.stroke, ink_texture=args.ink,
                            letter_tremor=args.tremor)

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
    scan_fn = None
    if args.scan > 0:
        from hw.paper import scan_effect
        base_seed = args.seed if args.seed is not None else 0
        scan_fn = lambda img, i: scan_effect(img, strength=args.scan, seed=base_seed + i)

    if args.toc is not None:
        # buffered path: render_document_with_toc needs the whole document
        # laid out once before it knows which page each heading landed on,
        # so this trades away the streaming property on purpose (see --toc's
        # help text) -- on_block/on_error still fire the same way underneath
        toc_pages, content_pages = r.render_document_with_toc(
            blocks, toc_title=args.toc, xh=args.xh, page_w=args.width, margin=MARGIN,
            ruled=args.ruled, slant=args.slant)
        if args.max_pages:
            content_pages = content_pages[:args.max_pages]
        for p in toc_pages:
            if args.title:
                r.stamp_header_footer(p, margin=MARGIN, xh=args.xh, title=args.title)
        for i, p in enumerate(content_pages, 1):
            if args.title or not args.no_page_numbers:
                r.stamp_header_footer(p, margin=MARGIN, xh=args.xh, title=args.title,
                                      page_num=None if args.no_page_numbers else i)
        all_pages = toc_pages + content_pages
        if scan_fn:
            all_pages = [scan_fn(p, i) for i, p in enumerate(all_pages)]
        for pi, page in enumerate(all_pages, 1):
            pp = f"{args.out}_p{pi}.png"
            page.save(pp)
            paths.append(pp)
            print(f"wrote {pp} ({page.size[0]}x{page.size[1]}) at {time.time() - t0:.1f}s",
                 flush=True)
        print(f"rendered {len(paths)} page(s) ({len(toc_pages)} table-of-contents) "
             f"in {time.time() - t0:.1f}s", flush=True)
    else:
        gen = r.iter_document(blocks, xh=args.xh, page_w=args.width, margin=MARGIN,
                              ruled=args.ruled, slant=args.slant, on_block=on_block,
                              on_error=on_error)
        for pi, page in enumerate(gen, 1):
            if args.title or not args.no_page_numbers:
                # stamped BEFORE the paper-scan warp, so the title/page number
                # distorts along with the rest of the page instead of looking
                # like a crisp overlay pasted onto a warped scan
                r.stamp_header_footer(page, margin=MARGIN, xh=args.xh, title=args.title,
                                      page_num=None if args.no_page_numbers else pi)
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
