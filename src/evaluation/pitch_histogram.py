# src/evaluation/pitch_histogram.py
"""
Pitch Histogram Analysis
========================
Computes and visualises pitch-class distributions.
Compares model outputs vs reference (real MAESTRO data).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import PLOTS_DIR, PIANO_ROLL_DIR
from src.evaluation.metrics import pitch_histogram

PITCH_NAMES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]


def plot_pitch_histograms(rolls_dict: dict, out_dir=PLOTS_DIR):
    """
    Plot side-by-side pitch-class histograms for multiple models.

    Args:
        rolls_dict: {"Model Name": np.ndarray(T, 88) or (N, T, 88)}
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(rolls_dict)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1: axes = [axes]

    for ax, (name, roll) in zip(axes, rolls_dict.items()):
        if roll.ndim == 3:   # (N, T, 88) → mean
            roll = roll.mean(axis=0)
        hist = pitch_histogram(roll)
        bars = ax.bar(range(12), hist, color="#4C72B0", alpha=0.8,
                       edgecolor="white")
        ax.set_xticks(range(12))
        ax.set_xticklabels(PITCH_NAMES, fontsize=8)
        ax.set_title(name, fontsize=9); ax.set_ylabel("Normalised Freq")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Pitch Class Distribution Comparison", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "pitch_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[pitch_histogram] → {path}")


def compare_pitch_sim(model_rolls: np.ndarray,
                        ref_path=None) -> float:
    """Compare model pitch histogram against reference MAESTRO data."""
    if ref_path is None:
        ref_path = os.path.join(PIANO_ROLL_DIR, "pianorolls.npy")
    if not os.path.exists(ref_path):
        print(f"[WARN] No reference data: {ref_path}")
        return float("nan")

    ref = np.load(ref_path)
    from src.evaluation.metrics import pitch_histogram_similarity
    sims = [pitch_histogram_similarity(model_rolls[i % len(model_rolls)],
                                        ref[i % len(ref)])
            for i in range(min(50, len(model_rolls)))]
    avg  = float(np.mean(sims))
    print(f"[pitch_histogram] Avg pitch sim (lower=closer): {avg:.4f}")
    return avg


if __name__ == "__main__":
    ref_path = os.path.join(PIANO_ROLL_DIR, "pianorolls.npy")
    if os.path.exists(ref_path):
        data = np.load(ref_path)[:100]
        plot_pitch_histograms({"MAESTRO (real)": data.mean(0)})
    else:
        print("Run preprocessing first.")