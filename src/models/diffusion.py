
# src/models/diffusion.py
"""
(Optional) Diffusion Model Stub
=================================
Placeholder for a score-based / DDPM approach to piano-roll generation.
Can be developed as an extension beyond the 4 required tasks.

Reference: Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
           Kong et al. (2021) "DiffWave"
"""

import torch
import torch.nn as nn
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import N_PITCHES, SEQ_LEN


class SinusoidalTimestepEmbedding(nn.Module):
    """Embed diffusion timestep t into a continuous vector."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb   # (B, dim)


class DiffusionUNet(nn.Module):
    """
    Minimal U-Net denoising network for piano-roll diffusion.
    Input : noisy piano-roll x_t  (B, SEQ_LEN, N_PITCHES)
            diffusion timestep  t (B,)
    Output: predicted noise ε̂    (B, SEQ_LEN, N_PITCHES)
    """

    def __init__(self, in_dim=N_PITCHES, seq_len=SEQ_LEN,
                 hidden=256, t_emb_dim=64, n_steps=1000):
        super().__init__()
        self.n_steps   = n_steps
        self.t_emb     = SinusoidalTimestepEmbedding(t_emb_dim)
        self.t_proj    = nn.Linear(t_emb_dim, hidden)

        self.enc1      = nn.Linear(in_dim, hidden)
        self.enc2      = nn.Linear(hidden, hidden)
        self.dec1      = nn.Linear(hidden, hidden)
        self.dec2      = nn.Linear(hidden, in_dim)
        self.act       = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.act(self.t_proj(self.t_emb(t)))   # (B, hidden)
        h     = self.act(self.enc1(x))                   # (B, T, hidden)
        h     = h + t_emb.unsqueeze(1)
        h     = self.act(self.enc2(h))
        h     = self.act(self.dec1(h))
        return self.dec2(h)   # (B, T, N_PITCHES)


class DDPM:
    """
    Denoising Diffusion Probabilistic Model wrapper.
    Manages forward (noising) and reverse (denoising) processes.
    """

    def __init__(self, model: DiffusionUNet, n_steps: int = 1000,
                 beta_start: float = 1e-4, beta_end: float = 0.02):
        self.model   = model
        self.n_steps = n_steps
        betas        = torch.linspace(beta_start, beta_end, n_steps)
        alphas       = 1.0 - betas
        self.register_schedule(betas, alphas)

    def register_schedule(self, betas, alphas):
        self.betas            = betas
        self.alphas           = alphas
        self.alpha_bar        = torch.cumprod(alphas, dim=0)
        self.sqrt_ab          = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_ab = torch.sqrt(1 - self.alpha_bar)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                  noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I)"""
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab     = self.sqrt_ab[t].view(-1, 1, 1).to(x0.device)
        s_1m_ab  = self.sqrt_one_minus_ab[t].view(-1, 1, 1).to(x0.device)
        return s_ab * x0 + s_1m_ab * noise

    def p_losses(self, x0: torch.Tensor) -> torch.Tensor:
        """Training loss: predict noise from noisy x_t."""
        B     = x0.size(0)
        t     = torch.randint(0, self.n_steps, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t   = self.q_sample(x0, t, noise)
        pred  = self.model(x_t, t)
        return nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, shape, device="cpu") -> torch.Tensor:
        """Reverse diffusion: iteratively denoise from x_T ~ N(0,I)."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred    = self.model(x, t_batch)
            alpha   = self.alphas[t].to(device)
            alpha_b = self.alpha_bar[t].to(device)
            coeff   = (1 - alpha) / torch.sqrt(1 - alpha_b)
            x       = (1 / torch.sqrt(alpha)) * (x - coeff * pred)
            if t > 0:
                noise = torch.randn_like(x)
                x    += torch.sqrt(self.betas[t].to(device)) * noise
        return torch.sigmoid(x)