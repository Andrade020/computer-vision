"""
Build a clean handwriting dataset from the OCR-segmented glyphs.

Reads segmented/recognized/<label>/*.jpg, keeps only valid single-character
classes, cleans noise heuristically, normalizes every glyph, and writes:

  hw/data/train.npz     -> X (N,64,64) float32 in [0,1], y (N,) int64, classes list
  hw/data/glyph_bank.pkl-> {char: [tight uint8 crops ...]}  (real ink for rendering)
  hw/data/manifest.json -> stats

Uses only numpy + PIL (no cv2), so it runs anywhere torch runs here.
"""
import os
import io
import json
import pickle
import glob
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RECOG_DIR = os.path.join(ROOT, "segmented", "recognized")
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CANVAS = 64          # training image size
GLYPH_FIT = 54       # glyph fits within this box, centered on CANVAS
BANK_MAXSIDE = 96    # max side for stored real-ink crops (keeps aspect)

VALID_CHARS = set("abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                  "0123456789")


def folder_to_char(name):
    """Map an OCR folder name to a single canonical character, or None."""
    if name.startswith("_") and len(name) == 2 and name[1].isalpha():
        return name[1].upper()          # "_A" -> "A"
    if len(name) == 1 and name in VALID_CHARS:
        return name
    return None


def load_binary(path):
    """Load a glyph, return a bool foreground mask (True = ink) or None."""
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    a = np.asarray(im, dtype=np.uint8)
    if a.size == 0:
        return None
    fg = a > 127                        # segmentation stored ink as white
    if fg.mean() > 0.5:                 # polarity flipped -> invert
        fg = ~fg
    return fg


def tight_crop(fg):
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return fg[y0:y1, x0:x1]


def is_noise(crop):
    """Cheap reject: too small, too sparse, wrong aspect, or fragmented."""
    h, w = crop.shape
    if h < 6 or w < 4:
        return True
    if crop.sum() < 25:
        return True
    if h > 6 * w or w > 8 * h:          # implausible aspect
        return True
    _, ncomp = label(crop)
    if ncomp > 10:                      # OCR garbage shatters into many pieces
        return True
    return False


def stroke_width(crop):
    """Scale-invariant thickness: max distance-to-background / glyph size.
    Thin pen strokes ~0.05-0.18; filled OCR blobs ~0.25+."""
    d = distance_transform_edt(crop)
    return float(d.max()) / max(crop.shape)


def to_canvas(crop):
    """Center a tight crop on a CANVAS x CANVAS float image, aspect preserved."""
    h, w = crop.shape
    scale = GLYPH_FIT / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    g = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    g = np.asarray(g, dtype=np.float32) / 255.0
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    oy, ox = (CANVAS - nh) // 2, (CANVAS - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = g
    return canvas


def to_bank(crop):
    """Store a real-ink crop as uint8 (ink=255), scaled so max side <= BANK_MAXSIDE."""
    h, w = crop.shape
    scale = min(1.0, BANK_MAXSIDE / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    g = Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    return np.asarray(g, dtype=np.uint8)


def main():
    folders = sorted(d for d in os.listdir(RECOG_DIR)
                     if os.path.isdir(os.path.join(RECOG_DIR, d)))
    bank = {}          # char -> list of uint8 crops
    train_imgs = []
    train_labels = []
    per_class_raw = {}
    per_class_kept = {}

    # Pass 1: collect surviving crops + per-glyph metrics
    raw = {}   # ch -> list of (crop, ink_ratio, blobbiness)
    for folder in folders:
        ch = folder_to_char(folder)
        if ch is None:
            continue
        paths = glob.glob(os.path.join(RECOG_DIR, folder, "*.jpg"))
        per_class_raw[ch] = per_class_raw.get(ch, 0) + len(paths)
        for p in paths:
            fg = load_binary(p)
            if fg is None:
                continue
            crop = tight_crop(fg)
            if crop is None or is_noise(crop):
                continue
            raw.setdefault(ch, []).append((crop, float(crop.mean()), stroke_width(crop)))

    # Pass 2: reject filled OCR blobs (thick strokes) + per-class density outliers
    dropped = 0
    for ch, items in raw.items():
        inks = np.array([m for _, m, _ in items])
        med, mad = np.median(inks), np.median(np.abs(inks - np.median(inks))) + 1e-6
        ink_hi = min(0.55, med + 4.0 * 1.4826 * mad)   # robust upper bound
        for crop, ink, sw in items:
            if sw > 0.23 or ink > ink_hi:
                dropped += 1
                continue
            bank.setdefault(ch, []).append(to_bank(crop))
            train_imgs.append(to_canvas(crop))
            train_labels.append(ch)
            per_class_kept[ch] = per_class_kept.get(ch, 0) + 1
    print(f"Per-class outlier rejection dropped {dropped} glyphs")

    classes = sorted(bank.keys())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    X = np.stack(train_imgs).astype(np.float32)
    y = np.array([cls_to_idx[c] for c in train_labels], dtype=np.int64)

    np.savez_compressed(os.path.join(DATA_DIR, "train.npz"),
                        X=X, y=y, classes=np.array(classes))
    with open(os.path.join(DATA_DIR, "glyph_bank.pkl"), "wb") as f:
        pickle.dump({"classes": classes, "bank": bank,
                     "canvas": CANVAS}, f, protocol=4)

    manifest = {
        "num_classes": len(classes),
        "classes": classes,
        "total_glyphs": int(len(y)),
        "per_class_kept": {c: per_class_kept.get(c, 0) for c in classes},
        "per_class_raw": per_class_raw,
        "canvas": CANVAS,
    }
    with open(os.path.join(DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Classes ({len(classes)}): {''.join(classes)}")
    print(f"Total kept glyphs: {len(y)}")
    print("Top classes:", sorted(per_class_kept.items(),
                                  key=lambda kv: -kv[1])[:15])
    low = [c for c in classes if per_class_kept.get(c, 0) < 5]
    print(f"Low-data classes (<5): {''.join(low)}")


if __name__ == "__main__":
    main()
