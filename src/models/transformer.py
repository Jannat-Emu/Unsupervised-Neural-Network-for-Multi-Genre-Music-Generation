# src/models/transformer.py
"""
Task 3 – Decoder-only Transformer
===================================
Autoregressive token-level music generation.

  h_t = Emb(x_t) + PosEnc(t)
  p(X) = ∏_t  p(x_t | x_{<t})

Loss:
  L_TR = −Σ_t log p_θ(x_t | x_{<t})

Perplexity:
  PPL = exp(L_TR / T)
"""

import math, sys, os
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (VOCAB_SIZE, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
                         TRANS_D_MODEL, TRANS_NHEAD, TRANS_NUM_LAYERS,
                         TRANS_DIM_FF, TRANS_DROPOUT, TRANS_MAX_LEN)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=TRANS_D_MODEL,
                 max_len=TRANS_MAX_LEN,
                 dropout=TRANS_DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class MusicTransformer(nn.Module):
    """Decoder-only Transformer for autoregressive music generation."""

    def __init__(self, vocab_size=VOCAB_SIZE,
                 d_model=TRANS_D_MODEL,
                 nhead=TRANS_NHEAD,
                 num_layers=TRANS_NUM_LAYERS,
                 dim_ff=TRANS_DIM_FF,
                 dropout=TRANS_DROPOUT,
                 max_len=TRANS_MAX_LEN):
        super().__init__()
        self.d_model   = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers)
        self.fc_out      = nn.Linear(d_model, vocab_size)

        nn.init.xavier_uniform_(self.token_emb.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _causal_mask(sz, device):
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(self, tokens):
        """
        tokens: (B, T) int
        returns logits: (B, T, vocab_size)
        """
        B, T = tokens.shape
        h    = self.token_emb(tokens) * math.sqrt(self.d_model)
        h    = self.pos_enc(h)
        mask = self._causal_mask(T, tokens.device)
        mem  = torch.zeros(B, 1, self.d_model, device=tokens.device)
        out  = self.transformer(h, mem, tgt_mask=mask,
                                 tgt_key_padding_mask=(tokens == PAD_TOKEN))
        return self.fc_out(out)

    @staticmethod
    def loss(logits, targets):
        """L_TR = cross-entropy, ignoring PAD tokens."""
        B, T, V = logits.shape
        return nn.functional.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T),
            ignore_index=PAD_TOKEN)

    @staticmethod
    def perplexity(avg_loss: float) -> float:
        return math.exp(min(avg_loss, 700))

    @torch.no_grad()
    def generate(self, max_len=256, temperature=1.0, device="cpu"):
        """Autoregressive sampling."""
        self.eval()
        tokens = torch.tensor([[BOS_TOKEN]], device=device)
        result = []
        for _ in range(max_len - 1):
            logits = self.forward(tokens)[:, -1, :] / temperature
            probs  = torch.softmax(logits, -1)
            nxt    = torch.multinomial(probs, 1).item()
            result.append(nxt)
            if nxt in (PAD_TOKEN, EOS_TOKEN):
                break
            tokens = torch.cat([tokens,
                                  torch.tensor([[nxt]], device=device)], dim=1)
        return result