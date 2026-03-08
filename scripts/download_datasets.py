#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║            MAESTRO Dataset Downloader & Setup Script                    ║
║  music-generation-unsupervised / download_dataset.py                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Downloads the MAESTRO v3.0.0 dataset (Classical Piano MIDI)            ║
║  Official page : https://magenta.tensorflow.org/datasets/maestro         ║
║  Direct link   : https://storage.googleapis.com/magentadata/            ║
║                  datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip        ║
║                                                                          ║
║  After downloading it:                                                   ║
║    1. Extracts all .mid files into   data/raw_midi/                      ║
║    2. Creates                        data/processed/  (empty, ready)     ║
║    3. Creates                        data/train_test_split/ (empty)      ║
║    4. Prints next steps                                                  ║
║                                                                          ║
║  Usage:                                                                  ║
║    python download_dataset.py                 # download + extract       ║
║    python download_dataset.py --check         # verify files only        ║
║    python download_dataset.py --info          # show manual steps        ║
║    python download_dataset.py --already_have  # skip download, just scan ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import shutil
import zipfile
import hashlib
import argparse
import urllib.request
from pathlib import Path


BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw_midi"
PROC_DIR    = DATA_DIR / "processed"
SPLIT_DIR   = DATA_DIR / "train_test_split"


MAESTRO = {
    "name"    : "MAESTRO v3.0.0",
    "url"     : "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
    "filename": "maestro-v3.0.0-midi.zip",
    "size_mb" : 57,               # MIDI-only zip is ~57 MB
    "page"    : "https://magenta.tensorflow.org/datasets/maestro",
    "paper"   : "https://arxiv.org/abs/1810.12247",
    "license" : "Creative Commons Attribution Non-Commercial Share-Alike 4.0",
    "tracks"  : 1276,
    "hours"   : 199,
    "genre"   : "Classical Piano",
}



_last_pct = -1

def _progress(count, block_size, total_size):
    global _last_pct
    if total_size <= 0:
        return
    pct = min(int(count * block_size * 100 / total_size), 100)
    if pct != _last_pct:
        filled = pct // 2
        bar    = "█" * filled + "░" * (50 - filled)
        mb_done = count * block_size / 1_048_576
        mb_tot  = total_size / 1_048_576
        sys.stdout.write(f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_tot:.1f} MB")
        sys.stdout.flush()
        _last_pct = pct
    if pct == 100:
        print()



def create_dirs():
    """Create all project data directories."""
    dirs = [
        RAW_DIR,
        PROC_DIR / "piano_rolls",
        PROC_DIR / "tokens",
        PROC_DIR / "metadata",
        SPLIT_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("[✓] Directory tree ready:")
    for d in dirs:
        print(f"    {d.relative_to(BASE_DIR)}/")



def download_maestro(dest_dir: Path = DATA_DIR) -> Path:
    """Download the MAESTRO MIDI zip file."""
    zip_path = dest_dir / MAESTRO["filename"]

    if zip_path.exists():
        print(f"[SKIP] Archive already exists: {zip_path}")
        return zip_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[↓] Downloading {MAESTRO['name']}")
    print(f"    URL  : {MAESTRO['url']}")
    print(f"    Size : ~{MAESTRO['size_mb']} MB  (MIDI-only package)")
    print(f"    Dest : {zip_path}\n")

    try:
        urllib.request.urlretrieve(MAESTRO["url"], zip_path, reporthook=_progress)
        print(f"\n[✓] Download complete → {zip_path}")
    except Exception as e:
        print(f"\n[✗] Download failed: {e}")
        print("\n── Manual Download Instructions ────────────────────────")
        print_manual_instructions()
        sys.exit(1)

    return zip_path



def extract_maestro(zip_path: Path, out_dir: Path = RAW_DIR) -> int:
    """
    Extract .mid files from the zip into data/raw_midi/.
    Flattens the directory structure (puts all .mid files directly in raw_midi/).
    Returns number of .mid files extracted.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[↗] Extracting .mid files → {out_dir}")

    count = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        midi_entries = [e for e in z.namelist()
                        if e.lower().endswith((".mid", ".midi"))]

        print(f"    Found {len(midi_entries)} MIDI files in archive")

        for entry in midi_entries:
            # Flatten: keep only the filename, not the nested path
            fname = Path(entry).name
            dest  = out_dir / fname
            # If name collision, keep subdirectory part to make unique
            if dest.exists():
                safe = entry.replace("/", "_").replace("\\", "_")
                dest = out_dir / safe

            with z.open(entry) as src, open(dest, "wb") as tgt:
                tgt.write(src.read())
            count += 1

            if count % 100 == 0:
                print(f"    ... {count}/{len(midi_entries)}")

    print(f"[✓] Extracted {count} MIDI files → {out_dir}")
    return count



def save_metadata(n_files: int):
    """Save dataset info as JSON for use by other scripts."""
    midi_files = sorted(RAW_DIR.glob("*.mid"))
    meta = {
        "dataset"      : MAESTRO["name"],
        "url"          : MAESTRO["url"],
        "license"      : MAESTRO["license"],
        "total_tracks" : n_files,
        "genre"        : MAESTRO["genre"],
        "hours"        : MAESTRO["hours"],
        "raw_midi_dir" : str(RAW_DIR),
        "files"        : [f.name for f in midi_files[:50]],  # first 50 as sample
    }
    meta_path = PROC_DIR / "metadata" / "dataset_info.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] Metadata saved → {meta_path.relative_to(BASE_DIR)}")


#

def verify(verbose: bool = True) -> dict:
    """Count and report on existing MIDI files."""
    mid_files = list(RAW_DIR.glob("*.mid")) + list(RAW_DIR.glob("*.midi"))
    total_kb  = sum(f.stat().st_size for f in mid_files) // 1024

    result = {
        "n_files"   : len(mid_files),
        "total_kb"  : total_kb,
        "raw_dir"   : str(RAW_DIR),
    }

    if verbose:
        print("\n── Dataset Verification ──────────────────────────────────")
        print(f"  MIDI files found : {len(mid_files)}")
        print(f"  Total size       : {total_kb / 1024:.1f} MB")
        print(f"  Location         : {RAW_DIR}")
        if mid_files:
            print(f"  Sample files     :")
            for f in mid_files[:5]:
                print(f"    {f.name}")
            if len(mid_files) > 5:
                print(f"    ... and {len(mid_files)-5} more")
        else:
            print("  [!] No MIDI files found — run without --check first.")
        print("──────────────────────────────────────────────────────────")

    return result



def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD — MAESTRO v3.0.0 (MIDI only)                          ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Step 1: Open this URL in your browser:                                  ║
║  https://magenta.tensorflow.org/datasets/maestro                         ║
║                                                                          ║
║  Step 2: Click "MIDI only" under "maestro-v3.0.0"                       ║
║    Direct link:                                                          ║
║    https://storage.googleapis.com/magentadata/datasets/maestro/          ║
║    v3.0.0/maestro-v3.0.0-midi.zip                                        ║
║                                                                          ║
║  Step 3: Save the file as:                                               ║
║    data/maestro-v3.0.0-midi.zip                                          ║
║    (inside your project folder)                                          ║
║                                                                          ║
║  Step 4: Run this script again:                                          ║
║    python download_dataset.py                                            ║
║                                                                          ║
║  Alternative — wget / curl:                                              ║
║    wget -O data/maestro-v3.0.0-midi.zip \                               ║
║      https://storage.googleapis.com/magentadata/datasets/maestro/        ║
║      v3.0.0/maestro-v3.0.0-midi.zip                                      ║
║                                                                          ║
║    curl -L -o data/maestro-v3.0.0-midi.zip \                            ║
║      https://storage.googleapis.com/magentadata/datasets/maestro/        ║
║      v3.0.0/maestro-v3.0.0-midi.zip                                      ║
║                                                                          ║
║  Expected after extraction:                                              ║
║    data/raw_midi/          ← ~1276 .mid files (~57 MB)                  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def print_next_steps():
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  ✓  MAESTRO dataset ready!  Next steps:                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  1. Preprocess MIDI → piano-rolls + tokens:                              ║
║       python src/preprocessing/midi_parser.py                            ║
║       python src/preprocessing/piano_roll.py                             ║
║       python src/preprocessing/tokenizer.py                              ║
║                                                                          ║
║  2. Split data into train / val / test:                                  ║
║       python src/preprocessing/midi_parser.py --split                   ║
║                                                                          ║
║  3. Run models:                                                          ║
║       python src/training/train_ae.py          # Task 1 – Autoencoder   ║
║       python src/training/train_vae.py         # Task 2 – VAE           ║
║       python src/training/train_transformer.py # Task 3 – Transformer   ║
║       python src/training/train_rl.py          # Task 4 – RLHF          ║
║                                                                          ║
║  4. Evaluate:                                                            ║
║       python src/evaluation/metrics.py                                   ║
║                                                                          ║
║  Data layout now:                                                        ║
║    data/raw_midi/          ← .mid files (source)                        ║
║    data/processed/         ← piano_rolls/, tokens/, metadata/           ║
║    data/train_test_split/  ← train.txt, val.txt, test.txt               ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download and set up MAESTRO v3.0.0 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--check",        action="store_true",
                        help="Verify existing files without downloading")
    parser.add_argument("--info",         action="store_true",
                        help="Show manual download instructions")
    parser.add_argument("--already_have", action="store_true",
                        help="Skip download — zip is already in data/ folder")
    args = parser.parse_args()

    # ── Banner ──────────────────────────────────────────────────────────
    print("=" * 68)
    print("  MAESTRO v3.0.0 Dataset Setup")
    print(f"  {MAESTRO['tracks']} tracks · {MAESTRO['hours']}h · {MAESTRO['genre']}")
    print(f"  {MAESTRO['page']}")
    print("=" * 68)

    if args.info:
        print_manual_instructions()
        return

    if args.check:
        verify()
        return

  
    create_dirs()

   
    if args.already_have:
        # User placed the zip manually somewhere
        candidates = list(DATA_DIR.glob("*.zip")) + list(BASE_DIR.glob("*.zip"))
        if not candidates:
            print("[✗] No .zip file found in data/ or project root.")
            print_manual_instructions()
            sys.exit(1)
        zip_path = candidates[0]
        print(f"[✓] Found existing archive: {zip_path}")
    else:
        zip_path = download_maestro(dest_dir=DATA_DIR)

   
    n = extract_maestro(zip_path, out_dir=RAW_DIR)

 
    answer = input("\n[?] Delete the zip archive to save space? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        zip_path.unlink()
        print(f"[✓] Deleted {zip_path.name}")

    
    save_metadata(n)
    verify()
    print_next_steps()


if __name__ == "__main__":
    main()