# src/models/vae.py
"""
Task 2 – Variational Autoencoder (β-VAE)
==========================================
Architecture:
  Encoder: BiLSTM → μ(X), log σ²(X)
  Sample:  z = μ + σ ⊙ ε,  ε ~ N(0,I)   [reparameterisation]
  Decoder: LSTM(z) → x̂

Loss:
  L_VAE  = L_recon + β · D_KL
  L_recon = ‖X − X̂‖²
  D_KL    = −½ Σ(1 + log σ² − μ² − σ²)
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (N_PITCHES, SEQ_LEN,
                         VAE_HIDDEN_DIM, VAE_LATENT_DIM,
                         VAE_NUM_LAYERS, VAE_BETA)


class VAEEncoder(nn.Module):
    def __init__(self, input_dim=N_PITCHES,
                 hidden_dim=VAE_HIDDEN_DIM,
                 latent_dim=VAE_LATENT_DIM,
                 num_layers=VAE_NUM_LAYERS):
        super().__init__()
        self.lstm   = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, bidirectional=True,
                               dropout=0.3 if num_layers > 1 else 0.0)
        self.fc_mu  = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_lv  = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.fc_mu(h), self.fc_lv(h)


class VAEDecoder(nn.Module):
    def __init__(self, output_dim=N_PITCHES,
                 hidden_dim=VAE_HIDDEN_DIM,
                 latent_dim=VAE_LATENT_DIM,
                 num_layers=VAE_NUM_LAYERS,
                 seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.z_to_h0    = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_c0    = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm       = nn.LSTM(latent_dim, hidden_dim, num_layers,
                                   batch_first=True,
                                   dropout=0.3 if num_layers > 1 else 0.0)
        self.out_fc     = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        B   = z.size(0)
        inp = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        h0  = (self.z_to_h0(z).view(B, self.num_layers, self.hidden_dim)
               .permute(1, 0, 2).contiguous())
        c0  = (self.z_to_c0(z).view(B, self.num_layers, self.hidden_dim)
               .permute(1, 0, 2).contiguous())
        out, _ = self.lstm(inp, (h0, c0))
        return torch.sigmoid(self.out_fc(out))


class MusicVAE(nn.Module):
    """β-VAE for multi-genre piano-roll generation."""

    def __init__(self, input_dim=N_PITCHES, beta=VAE_BETA, **kwargs):
        super().__init__()
        self.beta    = beta
        self.encoder = VAEEncoder(input_dim=input_dim, **kwargs)
        self.decoder = VAEDecoder(output_dim=input_dim, **kwargs)

    @staticmethod
    def reparameterise(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z           = self.reparameterise(mu, log_var)
        x_hat       = self.decoder(z)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        """
        L_VAE  = L_recon + β · D_KL
        D_KL   = −½ Σ(1 + log σ² − μ² − σ²)
        """
        l_recon = nn.functional.mse_loss(x_hat, x)
        kl      = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return l_recon + self.beta * kl, l_recon, kl

    @torch.no_grad()
    def generate(self, n=1, device="cpu"):
        self.eval()
        z = torch.randn(n, self.encoder.fc_mu.out_features, device=device)
        return self.decoder(z)

    @torch.no_grad()
    def interpolate(self, x1, x2, steps=8):
        """Linearly interpolate between two sequences in latent space."""
        self.eval()
        mu1, _ = self.encoder(x1)
        mu2, _ = self.encoder(x2)
        return torch.stack([
            self.decoder((1 - a) * mu1 + a * mu2)
            for a in torch.linspace(0, 1, steps)
        ])