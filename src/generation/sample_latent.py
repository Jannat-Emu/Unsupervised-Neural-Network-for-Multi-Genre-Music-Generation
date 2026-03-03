# src/generation/sample_latent.py
"""
Latent Space Exploration (VAE)
================================
Visualises and samples from the VAE latent space.
  - PCA / t-SNE 2D scatter of latent codes
  - Linear interpolation between two points
  - Random sampling with temperature control

Run:
  python src/generation/sample_latent.py --mode scatter
  python src/generation/sample_latent.py --mode interpolate
  python src/generation/sample_latent.py --mode sample --n 10
"""

import os, sys, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (CKPT_VAE, PIANO_ROLL_DIR, PLOTS_DIR,
                         MIDI_OUT_DIR, DEVICE, N_PITCHES)
from src.models.vae import MusicVAE
from src.generation.midi_export import pianoroll_to_midi


def load_vae(device=DEVICE):
    ckpt  = torch.load(CKPT_VAE, map_location=device)
    model = MusicVAE(input_dim=ckpt["input_dim"], beta=ckpt["beta"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["input_dim"]


def scatter_latent(model, data_path=None, method="tsne",
                    out_dir=PLOTS_DIR, device=DEVICE):
    """
    Encode real piano-rolls → latent codes → 2D scatter (PCA or t-SNE).
    """
    if data_path is None:
        data_path = os.path.join(PIANO_ROLL_DIR, "pianorolls.npy")
    if not os.path.exists(data_path):
        print(f"[WARN] Piano-roll data not found: {data_path}")
        return

    data  = torch.tensor(np.load(data_path).astype(np.float32))
    n_use = min(500, len(data))
    data  = data[:n_use].to(device)

    with torch.no_grad():
        mu, _ = model.encoder(data)
    z = mu.cpu().numpy()

    if method == "tsne":
        from sklearn.manifold import TSNE
        z2 = TSNE(n_components=2, random_state=42,
                   perplexity=min(30, n_use // 4)).fit_transform(z)
    else:
        from sklearn.decomposition import PCA
        z2 = PCA(n_components=2).fit_transform(z)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(z2[:, 0], z2[:, 1], s=8, alpha=0.6, c="#4C72B0")
    ax.set(title=f"VAE Latent Space ({method.upper()})",
           xlabel="Dim 1", ylabel="Dim 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"latent_scatter_{method}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[sample_latent] Scatter plot → {path}")


def interpolate_latent(model, data_path=None, steps=8,
                         out_dir=MIDI_OUT_DIR, device=DEVICE):
    """Generate MIDI by interpolating between two real encodings."""
    if data_path is None:
        data_path = os.path.join(PIANO_ROLL_DIR, "pianorolls.npy")
    if not os.path.exists(data_path):
        print(f"[WARN] {data_path} not found."); return

    data = torch.tensor(np.load(data_path).astype(np.float32))
    x1   = data[:1].to(device)
    x2   = data[-1:].to(device)
    interp = model.interpolate(x1, x2, steps=steps)   # (steps, 1, T, D)
    os.makedirs(out_dir, exist_ok=True)
    for i in range(steps):
        roll = interp[i, 0].detach().cpu().numpy()
        path = os.path.join(out_dir, f"latent_interp_{i+1:02d}.mid")
        pianoroll_to_midi(roll, path)
        print(f"  {path}")
    print(f"[sample_latent] {steps} interpolation MIDIs saved.")


def sample_random(model, n=8, temperature=1.0,
                   out_dir=MIDI_OUT_DIR, device=DEVICE):
    """Sample random z ~ N(0, temperature²·I) → decode → MIDI."""
    latent_dim = model.encoder.fc_mu.out_features
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        z     = torch.randn(n, latent_dim, device=device) * temperature
        rolls = model.decoder(z).cpu().numpy()
    for i, roll in enumerate(rolls):
        path = os.path.join(out_dir, f"latent_sample_{i+1:02d}.mid")
        pianoroll_to_midi(roll, path)
        print(f"  {path}")
    print(f"[sample_latent] {n} random samples saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="sample",
                   choices=["scatter", "interpolate", "sample"])
    p.add_argument("--n",           type=int,   default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--method",      type=str,   default="tsne",
                   choices=["tsne", "pca"])
    p.add_argument("--device",      type=str,   default=DEVICE)
    args = p.parse_args()

    model, _ = load_vae(args.device)

    if args.mode == "scatter":
        scatter_latent(model, method=args.method, device=args.device)
    elif args.mode == "interpolate":
        interpolate_latent(model, device=args.device)
    elif args.mode == "sample":
        sample_random(model, n=args.n, temperature=args.temperature,
                       device=args.device)