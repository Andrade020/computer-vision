"""
Conditional VAE glyph generator (the neural core).

Given a character class + a random style vector z ~ N(0, I), the decoder
produces a 64x64 glyph. Different z -> different handwriting instance of the
same letter, which is exactly the "writes anything, always a little different"
behaviour we want. Small enough to train on CPU overnight.
"""
import numpy as np
import torch
import torch.nn as nn

CANVAS = 64


class ConditionalVAE(nn.Module):
    def __init__(self, num_classes, latent_dim=64, embed_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),      # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),   # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True), # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),# 4x4
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim + embed_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, labels):
        emb = self.label_embedding(labels)
        h = self.fc_decode(torch.cat([z, emb], dim=1))
        return self.decoder(h.view(-1, 256, 4, 4))

    def forward(self, x, labels):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar


class GlyphGenerator:
    """Inference wrapper the renderer can use as a fallback source."""
    def __init__(self, ckpt_path, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.classes = list(ckpt["classes"])
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.model = ConditionalVAE(len(self.classes),
                                    latent_dim=ckpt.get("latent_dim", 64))
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def generate(self, ch, npr=None, temperature=0.7):
        """Return a tight float mask (H,W) in [0,1] for character ch.

        temperature < 1 scales the latent toward the prior mean, which makes a
        VAE emit sharper, more legible glyphs (less prior-sampling noise) while
        still varying per call; 1.0 = full-variance sampling.
        """
        if ch not in self.cls_to_idx:
            raise KeyError(ch)
        if npr is not None:
            z = torch.from_numpy(
                npr.randn(1, self.model.latent_dim).astype("float32"))
        else:
            z = torch.randn(1, self.model.latent_dim)
        z = z * temperature
        lab = torch.tensor([self.cls_to_idx[ch]], dtype=torch.long)
        out = self.model.decode(z, lab)[0, 0].cpu().numpy()
        return _tight(out)


def _tight(mask, thr=0.35):
    ink = mask > thr
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return mask
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return mask[y0:y1, x0:x1]
