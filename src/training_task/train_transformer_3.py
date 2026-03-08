# src/training/train_transformer.py
"""
Task 3 Training – Transformer Decoder
=======================================
Algorithm 3 (from paper):
  for epoch in E:
    for sequence X:
      for t = 1 to T: predict p(x_t | x_{<t})
      L_TR = −Σ log p
      θ ← θ − η∇L_TR

Outputs:
  checkpoints/checkpoint_transformer.pt
  outputs/plots/task3_transformer.png   (loss + perplexity)
  outputs/generated_midis/task3_trans_sample_*.mid  (10 samples)
"""

import os, sys, argparse, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (DEVICE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
                         GRAD_CLIP, TOKEN_DIR, PLOTS_DIR, CKPT_TRANS,
                         PAD_TOKEN, N_GEN_TRANS)
from src.models.transformer import MusicTransformer
from src.generation.generate_music import tokens_to_roll
from src.generation.midi_export import pianoroll_to_midi


class TokenDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = torch.tensor(tokens, dtype=torch.long)
    def __len__(self):  return len(self.tokens)
    def __getitem__(self, i):
        s = self.tokens[i]
        return s[:-1], s[1:]   # (src, tgt) teacher-forcing


def load_data(batch_size=BATCH_SIZE):
    path = os.path.join(TOKEN_DIR, "tokens_all.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Token dataset not found: {path}\n"
            "Run: python src/preprocessing/tokenizer.py")
    tokens = np.load(path)
    ds     = TokenDataset(tokens)
    n_val  = int(len(ds) * 0.15)
    tr, va = random_split(ds, [len(ds) - n_val, n_val])
    return DataLoader(tr, batch_size=batch_size, shuffle=True), \
           DataLoader(va, batch_size=batch_size)


def train(args):
    device   = args.device
    tr_dl, va_dl = load_data(args.batch_size)
    model    = MusicTransformer().to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=WEIGHT_DECAY)
    sched    = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 10,
        steps_per_epoch=len(tr_dl), epochs=args.epochs)

    print(f"[Task 3] Params: {sum(p.numel() for p in model.parameters()):,}")
    train_losses, val_losses, ppls = [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train(); ep = 0.0
        for src, tgt in tr_dl:
            src, tgt = src.to(device), tgt.to(device)
            logits   = model(src)
            loss     = model.loss(logits, tgt)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step(); sched.step(); ep += loss.item()
        train_losses.append(ep / len(tr_dl))

        model.eval(); vl = 0.0
        with torch.no_grad():
            for src, tgt in va_dl:
                src, tgt = src.to(device), tgt.to(device)
                vl += model.loss(model(src), tgt).item()
        vl /= len(va_dl)
        val_losses.append(vl)
        ppl = model.perplexity(vl)
        ppls.append(ppl)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{args.epochs}  "
                  f"Train={train_losses[-1]:.4f}  Val={vl:.4f}  PPL={ppl:.2f}")

    # Save
    os.makedirs(os.path.dirname(CKPT_TRANS), exist_ok=True)
    torch.save({"model_state"   : model.state_dict(),
                "train_losses"  : train_losses,
                "val_losses"    : val_losses,
                "ppls"          : ppls}, CKPT_TRANS)
    print(f"[Task 3] Checkpoint → {CKPT_TRANS}")

    _plot(train_losses, val_losses, ppls)
    _print_report(val_losses, ppls)

    # Generate
    for i in range(N_GEN_TRANS):
        toks = model.generate(max_len=512, temperature=args.temperature,
                               device=device)
        roll = tokens_to_roll(toks)
        out  = os.path.join(args.out_dir, f"task3_trans_sample_{i+1:02d}.mid")
        pianoroll_to_midi(roll, out)
        print(f"  Generated: {out}")

    return model, ppls


def _plot(train, val, ppls):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train, label="Train"); ax1.plot(val, label="Val", ls="--")
    ax1.set(title="Task 3 – Transformer Loss",
            xlabel="Epoch", ylabel="Cross-Entropy Loss")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ppls, color="#55A868")
    ax2.set(title="Task 3 – Perplexity", xlabel="Epoch", ylabel="PPL")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "task3_transformer.png")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Task 3] Plot → {path}")


def _print_report(val, ppls):
    best = min(ppls); best_ep = ppls.index(best) + 1
    print("\n" + "=" * 50)
    print("PERPLEXITY EVALUATION REPORT")
    print("=" * 50)
    print(f"  Final Val Loss   : {val[-1]:.4f}")
    print(f"  Final PPL        : {ppls[-1]:.2f}")
    print(f"  Best PPL         : {best:.2f}  (epoch {best_ep})")
    print(f"  Paper target PPL : ~12.5")
    print("=" * 50)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",      type=int,   default=EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device",      type=str,   default=DEVICE)
    p.add_argument("--out_dir",     type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "outputs", "generated_midis"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*55}\nTask 3 – Transformer  |  device={args.device}\n{'='*55}")
    train(args)
    print("✓ Task 3 complete.")