# src/generation/generate_music.py
"""
Music Generation Utilities
============================
Helpers shared by all training scripts:
  - tokens_to_roll(): convert token list → piano-roll array
  - generate_from_checkpoint(): load model + generate N MIDIs
  - CLI for quick generation from any checkpoint
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (SEQ_LEN, N_PITCHES, PITCH_MIN,
                         PITCH_OFFSET, VEL_OFFSET, DUR_OFFSET,
                         N_VEL_BINS, N_DUR_BINS,
                         BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
                         DEVICE, MIDI_OUT_DIR,
                         CKPT_AE, CKPT_VAE, CKPT_TRANS, CKPT_RLHF)
from src.generation.midi_export import pianoroll_to_midi


def tokens_to_roll(toks, T=SEQ_LEN, P=N_PITCHES):
    """
    Convert a flat token list → approximate (T, 88) piano-roll.
    Groups pitch/velocity/duration triples and fills roll cells.
    """
    roll  = np.zeros((T, P), dtype=np.float32)
    t_idx = 0
    j     = 0
    while j < len(toks) - 2 and t_idx < T:
        tok = int(toks[j])
        if PITCH_OFFSET <= tok < VEL_OFFSET:
            p_idx   = tok - PITCH_OFFSET
            vel_tok = int(toks[j + 1]) if j + 1 < len(toks) else VEL_OFFSET
            dur_tok = int(toks[j + 2]) if j + 2 < len(toks) else DUR_OFFSET
            if VEL_OFFSET <= vel_tok < DUR_OFFSET:
                vel   = (vel_tok - VEL_OFFSET) / N_VEL_BINS
                steps = max(1, dur_tok - DUR_OFFSET)
                for dt in range(min(steps, T - t_idx)):
                    if 0 <= p_idx < P:
                        roll[t_idx + dt, p_idx] = vel
                t_idx += max(1, steps // 4)
                j += 3; continue
        j += 1
    return roll


def generate_from_ae(n=5, device=DEVICE, out_dir=MIDI_OUT_DIR):
    from src.models.autoencoder import LSTMAutoencoder
    ckpt  = torch.load(CKPT_AE, map_location=device)
    # model = LSTMAutoencoder(input_dim=ckpt["input_dim"]).to(device)
    model = LSTMAutoencoder(input_dim=ckpt["input_dim"], latent_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    rolls = model.generate(n=n, device=device).cpu().numpy()
    paths = []
    for i, roll in enumerate(rolls):
        p = os.path.join(out_dir, f"gen_ae_{i+1:02d}.mid")
        pianoroll_to_midi(roll, p); paths.append(p)
    return paths


def generate_from_vae(n=8, device=DEVICE, out_dir=MIDI_OUT_DIR):
    from src.models.vae import MusicVAE
    ckpt  = torch.load(CKPT_VAE, map_location=device)
    model = MusicVAE(input_dim=ckpt["input_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    rolls = model.generate(n=n, device=device).cpu().numpy()
    paths = []
    for i, roll in enumerate(rolls):
        p = os.path.join(out_dir, f"gen_vae_{i+1:02d}.mid")
        pianoroll_to_midi(roll, p); paths.append(p)
    return paths


def generate_from_transformer(n=10, temperature=1.0,
                                device=DEVICE, out_dir=MIDI_OUT_DIR):
    from src.models.transformer import MusicTransformer
    ckpt  = torch.load(CKPT_TRANS, map_location=device)
    model = MusicTransformer().to(device)
    model.load_state_dict(ckpt["model_state"])
    paths = []
    for i in range(n):
        toks = model.generate(max_len=512, temperature=temperature,
                               device=device)
        roll = tokens_to_roll(toks)
        p    = os.path.join(out_dir, f"gen_trans_{i+1:02d}.mid")
        pianoroll_to_midi(roll, p); paths.append(p)
    return paths


def generate_from_rlhf(n=10, temperature=0.9,
                         device=DEVICE, out_dir=MIDI_OUT_DIR):
    from src.models.transformer import MusicTransformer
    ckpt  = torch.load(CKPT_RLHF, map_location=device)
    model = MusicTransformer().to(device)
    model.load_state_dict(ckpt["model_state"])
    paths = []
    for i in range(n):
        toks = model.generate(max_len=512, temperature=temperature,
                               device=device)
        roll = tokens_to_roll(toks)
        p    = os.path.join(out_dir, f"gen_rlhf_{i+1:02d}.mid")
        pianoroll_to_midi(roll, p); paths.append(p)
    return paths


GENERATORS = {
    "ae"   : generate_from_ae,
    "vae"  : generate_from_vae,
    "trans": generate_from_transformer,
    "rlhf" : generate_from_rlhf,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate MIDI from trained model")
    p.add_argument("--model",       choices=list(GENERATORS), default="trans")
    p.add_argument("--n",           type=int,   default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device",      type=str,   default=DEVICE)
    p.add_argument("--out_dir",     type=str,   default=MIDI_OUT_DIR)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fn = GENERATORS[args.model]
    kwargs = {"n": args.n, "device": args.device, "out_dir": args.out_dir}
    if args.model in ("trans", "rlhf"):
        kwargs["temperature"] = args.temperature
    paths = fn(**kwargs)
    for p in paths:
        print(f"  → {p}")