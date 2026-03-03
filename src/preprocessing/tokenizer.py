# src/preprocessing/tokenizer.py
"""
Event Tokenizer
================
Converts MIDI note events into flat integer token sequences and back.

Token layout  (VOCAB_SIZE = 139):
  0          PAD
  1          BOS  (begin of sequence)
  2          EOS  (end of sequence)
  3  – 90    pitch tokens  (MIDI pitch 21–108)
  91 – 122   velocity bins (32 bins)
  123 – 138  duration bins (16 bins)

Saves:
  data/processed/tokens/tokens_all.npy   (N, SEQ_LEN) int32
"""

import os, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (RAW_MIDI_DIR, TOKEN_DIR, META_DIR,
                         PITCH_MIN, PITCH_MAX,
                         PAD_TOKEN, BOS_TOKEN, EOS_TOKEN,
                         PITCH_OFFSET, VEL_OFFSET, DUR_OFFSET,
                         N_VEL_BINS, N_DUR_BINS, SEQ_LEN)

try:
    import pretty_midi
    HAS_PM = True
except ImportError:
    HAS_PM = False


# ── Quantise helpers ─────────────────────────────────────────────────

def _vel_bin(velocity: int) -> int:
    return min(int(velocity / 128 * N_VEL_BINS), N_VEL_BINS - 1)


def _dur_bin(duration_sec: float, tempo: float = 120.0) -> int:
    beats = duration_sec * tempo / 60.0
    b     = int(beats / 4.0 * N_DUR_BINS)
    return min(b, N_DUR_BINS - 1)


# ── Encode ───────────────────────────────────────────────────────────

def events_to_tokens(events: list, max_len: int = SEQ_LEN) -> np.ndarray:
    """
    Convert a list of note-event dicts to a token sequence.
    Format per note: [pitch_tok, vel_tok, dur_tok]
    Full seq:        [BOS, p0,v0,d0, p1,v1,d1, …, EOS, PAD…]
    Returns np.ndarray shape (max_len,) dtype int32.
    """
    toks = [BOS_TOKEN]
    for e in events:
        toks.append(PITCH_OFFSET + (e["pitch"] - PITCH_MIN))
        toks.append(VEL_OFFSET   + _vel_bin(e["velocity"]))
        toks.append(DUR_OFFSET   + _dur_bin(e["duration_sec"]))
        if len(toks) >= max_len - 1:
            break
    toks.append(EOS_TOKEN)
    if len(toks) < max_len:
        toks += [PAD_TOKEN] * (max_len - len(toks))
    return np.array(toks[:max_len], dtype=np.int32)


# ── Decode ───────────────────────────────────────────────────────────

def tokens_to_events(tokens: np.ndarray, tempo: float = 120.0) -> list:
    """Inverse: token sequence → list of note-event dicts (approx)."""
    events = []
    i = 0
    while i < len(tokens) - 2:
        tok = int(tokens[i])
        if tok in (PAD_TOKEN, EOS_TOKEN):
            break
        if tok == BOS_TOKEN:
            i += 1; continue
        if PITCH_OFFSET <= tok < VEL_OFFSET:
            pitch    = tok - PITCH_OFFSET + PITCH_MIN
            vel_tok  = int(tokens[i + 1]) if i + 1 < len(tokens) else VEL_OFFSET
            dur_tok  = int(tokens[i + 2]) if i + 2 < len(tokens) else DUR_OFFSET
            velocity = int((vel_tok - VEL_OFFSET) / N_VEL_BINS * 128)
            dur_sec  = (dur_tok - DUR_OFFSET) / N_DUR_BINS * 4.0 * 60.0 / tempo
            events.append({
                "pitch"       : pitch,
                "velocity"    : max(1, velocity),
                "start_sec"   : 0.0,
                "duration_sec": max(0.05, dur_sec),
            })
            i += 3
        else:
            i += 1
    return events


# ── Build full token dataset ──────────────────────────────────────────

def build_dataset(raw_dir: str = RAW_MIDI_DIR,
                  out_dir: str = TOKEN_DIR,
                  max_len: int = SEQ_LEN) -> np.ndarray:
    """
    Parse all MIDI files → token sequences → save as .npy.
    Returns array of shape (N, max_len).
    """
    if not HAS_PM:
        raise ImportError("pip install pretty_midi")

    os.makedirs(out_dir, exist_ok=True)
    midi_files = sorted(
        str(p) for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in (".mid", ".midi")
    )
    print(f"[tokenizer] {len(midi_files)} MIDI files")

    seqs = []
    for fp in tqdm(midi_files, desc="Tokenising"):
        pm     = pretty_midi.PrettyMIDI(fp)
        events = []
        for inst in pm.instruments:
            if inst.is_drum: continue
            for note in inst.notes:
                if PITCH_MIN <= note.pitch <= PITCH_MAX:
                    events.append({
                        "pitch"       : note.pitch,
                        "velocity"    : note.velocity,
                        "duration_sec": note.end - note.start,
                    })
        events.sort(key=lambda e: e.get("start_sec", 0))
        seq = events_to_tokens(events, max_len)
        if seq[seq != PAD_TOKEN].sum() > 0:
            seqs.append(seq)

    if not seqs:
        print("[WARN] No token sequences extracted.")
        return np.array([])

    arr = np.stack(seqs)   # (N, max_len)
    np.save(os.path.join(out_dir, "tokens_all.npy"), arr)
    print(f"[tokenizer] Token dataset: {arr.shape}  →  {out_dir}")
    return arr


if __name__ == "__main__":
    build_dataset()