
# src/preprocessing/piano_roll.py
"""
Piano Roll Converter
=====================
MIDI files → (SEQ_LEN, 88) float32 piano-roll arrays.
Saves:
  data/processed/piano_rolls/pianorolls.npy   (N, SEQ_LEN, 88)
  data/processed/piano_rolls/file_index.txt   source filenames
"""

import os, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (RAW_MIDI_DIR, PIANO_ROLL_DIR, SPLIT_DIR,
                         PITCH_MIN, PITCH_MAX, N_PITCHES,
                         SEQ_LEN, STEPS_PER_BEAT)

try:
    import pretty_midi
    HAS_PM = True
except ImportError:
    HAS_PM = False


def midi_to_pianoroll(path: str, n_steps: int = SEQ_LEN) -> np.ndarray:
    """
    Convert one MIDI file to a (n_steps, 88) float32 piano-roll.
    Values in [0, 1] representing velocity (0 = silent).
    """
    if not HAS_PM:
        raise ImportError("pip install pretty_midi")
    try:
        pm   = pretty_midi.PrettyMIDI(path)
        roll = pm.get_piano_roll(fs=STEPS_PER_BEAT)         # (128, T)
        roll = roll[PITCH_MIN:PITCH_MAX + 1, :].T / 127.0   # (T, 88)
    except Exception:
        return np.zeros((n_steps, N_PITCHES), dtype=np.float32)

    T = roll.shape[0]
    if T >= n_steps:
        return roll[:n_steps].astype(np.float32)
    pad = np.zeros((n_steps - T, N_PITCHES), dtype=np.float32)
    return np.vstack([roll, pad]).astype(np.float32)


def pianoroll_to_midi(roll: np.ndarray,
                       path: str,
                       tempo: float = 120.0,
                       threshold: float = 0.1):
    """Convert a (T, 88) piano-roll back to a MIDI file."""
    if not HAS_PM:
        raise ImportError("pip install pretty_midi")
    pm    = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    step_dur = 60.0 / tempo / STEPS_PER_BEAT
    T, P     = roll.shape
    active   = {}

    for t in range(T):
        for p in range(P):
            pitch = p + PITCH_MIN
            val   = float(roll[t, p])
            time  = t * step_dur
            if val >= threshold:
                if pitch not in active:
                    active[pitch] = (time, max(1, int(val * 127)))
            else:
                if pitch in active:
                    s, v = active.pop(pitch)
                    if time > s:
                        piano.notes.append(pretty_midi.Note(v, pitch, s, time))

    end = T * step_dur
    for pitch, (s, v) in active.items():
        piano.notes.append(pretty_midi.Note(v, pitch, s, end))

    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pm.write(path)


def build_dataset(raw_dir: str = RAW_MIDI_DIR,
                  out_dir: str = PIANO_ROLL_DIR,
                  n_steps: int = SEQ_LEN) -> tuple:
    """
    Walk raw_dir, convert every MIDI → piano-roll, save dataset.
    Returns (rolls, file_names) where rolls.shape = (N, n_steps, 88).
    """
    os.makedirs(out_dir, exist_ok=True)
    midi_files = sorted(
        str(p) for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in (".mid", ".midi")
    )
    print(f"[piano_roll] {len(midi_files)} MIDI files found")

    rolls, names = [], []
    for fp in tqdm(midi_files, desc="Piano-roll conversion"):
        roll = midi_to_pianoroll(fp, n_steps)
        if roll.sum() > 0:
            rolls.append(roll)
            names.append(Path(fp).name)

    if not rolls:
        print("[WARN] No rolls extracted.")
        return np.array([]), []

    arr = np.stack(rolls)   # (N, n_steps, 88)
    np.save(os.path.join(out_dir, "pianorolls.npy"), arr)
    with open(os.path.join(out_dir, "file_index.txt"), "w") as f:
        f.write("\n".join(names))
    print(f"[piano_roll] Dataset: {arr.shape}  →  {out_dir}")
    return arr, names


if __name__ == "__main__":
    build_dataset()
