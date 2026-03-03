# src/models/autoencoder.py
"""
Task 1 – LSTM Autoencoder
==========================
Architecture:
  Encoder : BiLSTM  →  linear  →  z ∈ ℝ^{64}
  Decoder : LSTM(z repeated T times)  →  linear  →  x̂

Loss:
  L_AE = (1/T) Σ_t ‖x_t − x̂_t‖²
"""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (N_PITCHES, SEQ_LEN,
                         AE_HIDDEN_DIM, AE_LATENT_DIM,
                         AE_NUM_LAYERS, AE_DROPOUT)


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=N_PITCHES,
                 hidden_dim=AE_HIDDEN_DIM,
                 latent_dim=AE_LATENT_DIM,
                 num_layers=AE_NUM_LAYERS,
                 dropout=AE_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(self, x):
        # x: (B, T, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers*2, B, hidden) — take last layer, both directions
        z = self.fc(torch.cat([h_n[-2], h_n[-1]], dim=-1))
        return z   # (B, latent_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim=N_PITCHES,
                 hidden_dim=AE_HIDDEN_DIM,
                 latent_dim=AE_LATENT_DIM,
                 num_layers=AE_NUM_LAYERS,
                 dropout=AE_DROPOUT,
                 seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.z_to_h0    = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_c0    = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm       = nn.LSTM(latent_dim, hidden_dim, num_layers,
                                   batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0.0)
        self.out_fc     = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        B   = z.size(0)
        inp = z.unsqueeze(1).repeat(1, self.seq_len, 1)   # (B, T, latent)
        h0  = (self.z_to_h0(z)
               .view(B, self.num_layers, self.hidden_dim)
               .permute(1, 0, 2).contiguous())
        c0  = (self.z_to_c0(z)
               .view(B, self.num_layers, self.hidden_dim)
               .permute(1, 0, 2).contiguous())
        out, _ = self.lstm(inp, (h0, c0))
        return torch.sigmoid(self.out_fc(out))   # (B, T, output_dim)


class LSTMAutoencoder(nn.Module):
    """Full LSTM Autoencoder for piano-roll reconstruction."""

    def __init__(self, input_dim=N_PITCHES, **kwargs):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim=input_dim, **kwargs)
        self.decoder = LSTMDecoder(output_dim=input_dim, **kwargs)

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    @torch.no_grad()
    def generate(self, n=1, device="cpu"):
        self.eval()
        latent_dim = self.encoder.fc.out_features
        z          = torch.randn(n, latent_dim, device=device)
        return self.decoder(z)   # (n, T, 88)

    @staticmethod
    def loss(x, x_hat):
        """L_AE = MSE(x̂, x)"""
        return nn.functional.mse_loss(x_hat, x)