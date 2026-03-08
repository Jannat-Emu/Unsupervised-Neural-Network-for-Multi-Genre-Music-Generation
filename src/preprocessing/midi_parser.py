# src/preprocessing/midi_parser.py
"""
MIDI Parser
============
Reads every .mid file in data/raw_midi/, extracts note events,
builds train / val / test file lists, and saves:
  data/processed/metadata/note_events_sample.json
  data/train_test_split/train.txt
  data/train_test_split/val.txt
  data/train_test_split/test.txt
"""

import os, sys, json, random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (RAW_MIDI_DIR, META_DIR, SPLIT_DIR,
                         PITCH_MIN, PITCH_MAX, TEST_SPLIT, VAL_SPLIT)

try:
    import pretty_midi
    HAS_PRETTY_MIDI = True
except ImportError:
    HAS_PRETTY_MIDI = False


def parse_midi(path: str) -> list:
    """
    Parse a single .mid file into a list of note-event dicts.
    Each dict: {pitch, velocity, start_sec, end_sec, duration_sec}
    Only non-drum notes in [PITCH_MIN, PITCH_MAX]. Returns [] on error.
    """
    if not HAS_PRETTY_MIDI:
        raise ImportError("pretty_midi not installed: pip install pretty_midi")
    try:
        pm = pretty_midi.PrettyMIDI(path)
    except Exception as e:
        print(f"  [WARN] Cannot parse {Path(path).name}: {e}")
        return []

    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if PITCH_MIN <= note.pitch <= PITCH_MAX:
                events.append({
                    "pitch"       : note.pitch,
                    "velocity"    : note.velocity,
                    "start_sec"   : round(note.start, 4),
                    "end_sec"     : round(note.end, 4),
                    "duration_sec": round(note.end - note.start, 4),
                })
    events.sort(key=lambda e: e["start_sec"])
    return events


def collect_midi_files(raw_dir: str = RAW_MIDI_DIR) -> list:
    files = sorted(
        str(p) for p in Path(raw_dir).rglob("*")
        if p.suffix.lower() in (".mid", ".midi")
    )
    print(f"[midi_parser] Found {len(files)} MIDI files in {raw_dir}")
    return files


def split_files(files, val_frac=VAL_SPLIT, test_frac=TEST_SPLIT, seed=42):
    random.seed(seed)
    shuffled = files[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test  = shuffled[:n_test]
    val   = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def save_split(train, val, test, out_dir=SPLIT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    for name, lst in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(lst))
        print(f"  [{name}] {len(lst)} files  →  {path}")


def load_split(split="train", split_dir=SPLIT_DIR):
    path = os.path.join(split_dir, f"{split}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split not found: {path}\n"
            "Run: python src/preprocessing/midi_parser.py")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def save_sample_metadata(files, n_sample=5):
    samples = {}
    for fp in files[:n_sample]:
        events = parse_midi(fp) if HAS_PRETTY_MIDI else []
        samples[Path(fp).name] = {"n_events": len(events), "first_10": events[:10]}
    os.makedirs(META_DIR, exist_ok=True)
    meta_path = os.path.join(META_DIR, "note_events_sample.json")
    with open(meta_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[midi_parser] Sample metadata → {meta_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no_split", action="store_true")
    args = p.parse_args()

    files = collect_midi_files()
    if not files:
        print("[ERROR] No MIDI files found. Run download_dataset.py first.")
        sys.exit(1)

    if not args.no_split:
        train, val, test = split_files(files)
        save_split(train, val, test)
        print(f"  Total={len(files)}  Train={len(train)}  Val={len(val)}  Test={len(test)}")

    save_sample_metadata(files)
    print("[midi_parser] Done.")