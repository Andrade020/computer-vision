"""
Train the Conditional VAE glyph generator on CPU, overnight-friendly.

Features:
  - class-balanced sampling (data is very unequal)
  - light on-the-fly augmentation (affine jitter)
  - KL annealing (beta warmup) for stable, non-blurry glyphs
  - checkpoint + sample grid every SAVE_EVERY epochs (resumable)
  - writes best/last checkpoints to hw/checkpoints/

Run:  python -m hw.train --epochs 4000
"""
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .model import ConditionalVAE, CANVAS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "train.npz")
CKPT_DIR = os.path.join(HERE, "checkpoints")
SAMPLE_DIR = os.path.join(HERE, "samples")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


class GlyphDS(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _aug(self, img):
        # small affine jitter via torch grid_sample
        t = torch.from_numpy(img)[None, None]
        ang = (np.random.rand() - 0.5) * 0.35          # +-10 deg
        sc = 1.0 + (np.random.rand() - 0.5) * 0.16
        tx = (np.random.rand() - 0.5) * 0.12
        ty = (np.random.rand() - 0.5) * 0.12
        cos, sin = np.cos(ang) / sc, np.sin(ang) / sc
        theta = torch.tensor([[[cos, -sin, tx], [sin, cos, ty]]], dtype=torch.float32)
        grid = F.affine_grid(theta, t.shape, align_corners=False)
        t = F.grid_sample(t, grid, align_corners=False, padding_mode="zeros")
        return t[0, 0].numpy()

    def __getitem__(self, i):
        img = self.X[i]
        if self.augment:
            img = self._aug(img)
        return torch.from_numpy(img)[None].float(), int(self.y[i])


def loss_fn(recon, x, mu, logvar, beta):
    bce = F.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kld, bce, kld


@torch.no_grad()
def save_grid(model, classes, epoch, device):
    from PIL import Image
    model.eval()
    n = len(classes)
    z = torch.randn(n, model.latent_dim, device=device)
    labs = torch.arange(n, device=device)
    out = model.decode(z, labs).cpu().numpy()[:, 0]
    cols = 13
    rows = (n + cols - 1) // cols
    sheet = np.ones((rows * CANVAS, cols * CANVAS), np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        sheet[r*CANVAS:(r+1)*CANVAS, c*CANVAS:(c+1)*CANVAS] = 1 - out[i]
    Image.fromarray((sheet*255).astype(np.uint8)).save(
        os.path.join(SAMPLE_DIR, f"epoch_{epoch:05d}.png"))
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1.2e-3)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--kl_warmup", type=int, default=400)
    ap.add_argument("--beta_max", type=float, default=0.6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    device = "cpu"

    d = np.load(DATA, allow_pickle=True)
    X, y, classes = d["X"], d["y"], list(d["classes"])
    print(f"Loaded {len(y)} glyphs, {len(classes)} classes on {device}")

    # class-balanced sampling weights
    counts = np.bincount(y, minlength=len(classes)).astype(np.float64)
    inv = 1.0 / np.clip(counts, 1, None)
    sample_w = inv[y]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(y), replacement=True)

    ds = GlyphDS(X, y, augment=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                    num_workers=0, drop_last=True)

    model = ConditionalVAE(len(classes), latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    start_epoch = 1
    best = float("inf")
    last_path = os.path.join(CKPT_DIR, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best = ck.get("best", best)
        print(f"Resumed from epoch {ck['epoch']}")

    log_path = os.path.join(CKPT_DIR, "train_log.csv")
    if start_epoch == 1:
        with open(log_path, "w") as f:
            f.write("epoch,loss,bce,kld,beta,sec\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        beta = args.beta_max * min(1.0, epoch / args.kl_warmup)
        model.train()
        tot = tb = tk = 0.0
        for xb, lb in dl:
            xb, lb = xb.to(device), lb.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(xb, lb)
            loss, bce, kld = loss_fn(recon, xb, mu, logvar, beta)
            loss.backward()
            opt.step()
            tot += loss.item(); tb += bce.item(); tk += kld.item()
        sched.step()
        n = len(ds)
        dt = time.time() - t0
        with open(log_path, "a") as f:
            f.write(f"{epoch},{tot/n:.4f},{tb/n:.4f},{tk/n:.4f},{beta:.3f},{dt:.1f}\n")

        if epoch % 10 == 0 or epoch == 1:
            print(f"ep {epoch:5d} loss {tot/n:8.3f} bce {tb/n:8.3f} "
                  f"kld {tk/n:7.3f} beta {beta:.2f} {dt:.1f}s")

        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "epoch": epoch, "classes": classes, "latent_dim": args.latent,
              "best": best}
        torch.save(ck, last_path)
        if tot / n < best:
            best = tot / n
            ck["best"] = best
            torch.save(ck, os.path.join(CKPT_DIR, "best.pt"))
        if epoch % args.save_every == 0 or epoch == 1:
            save_grid(model, classes, epoch, device)

    print("Training done. best loss", best)


if __name__ == "__main__":
    main()
