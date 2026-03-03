# src/evaluation/rhythm_score.py
"""
Rhythm Scoring & Visualisation
================================
Bar charts comparing rhythm diversity and repetition ratio
across all models (Table 3 comparison).
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (PLOTS_DIR, PIANO_ROLL_DIR,
                         CKPT_AE, CKPT_VAE, CKPT_TRANS, CKPT_RLHF,
                         DEVICE, N_GEN_AE, N_GEN_VAE)
from src.evaluation.metrics import (rhythm_diversity, repetition_ratio,
                                      note_density, evaluate_batch)


# Reference values from paper Table 3
PAPER_RESULTS = {
    "Random"      : {"rhythm_diversity": 0.12, "repetition_ratio": 0.75, "human_score": 1.1},
    "Markov Chain": {"rhythm_diversity": 0.31, "repetition_ratio": 0.58, "human_score": 2.3},
    "Task 1 AE"   : {"rhythm_diversity": 0.45, "repetition_ratio": 0.42, "human_score": 3.1},
    "Task 2 VAE"  : {"rhythm_diversity": 0.58, "repetition_ratio": 0.35, "human_score": 3.8},
    "Task 3 Trans": {"rhythm_diversity": 0.72, "repetition_ratio": 0.25, "human_score": 4.4},
    "Task 4 RLHF" : {"rhythm_diversity": 0.79, "repetition_ratio": 0.18, "human_score": 4.8},
}


def plot_rhythm_comparison(results=None, out_dir=PLOTS_DIR):
    """Bar chart: rhythm diversity + repetition ratio per model."""
    if results is None:
        results = PAPER_RESULTS

    models = list(results.keys())
    rd     = [results[m]["rhythm_diversity"] for m in models]
    rr     = [results[m]["repetition_ratio"] for m in models]

    x = np.arange(len(models))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(x, rd, color="#4C72B0", alpha=0.85, width=0.55)
    ax1.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax1.set(title="Rhythm Diversity ↑  (higher = more varied rhythm)",
            ylabel="Rhythm Diversity Score", xticks=x, xticklabels=models,
            ylim=(0, 1.0))
    ax1.tick_params(axis="x", rotation=20); ax1.grid(axis="y", alpha=0.3)

    bars = ax2.bar(x, rr, color="#DD8452", alpha=0.85, width=0.55)
    ax2.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax2.set(title="Repetition Ratio ↓  (lower = more creative)",
            ylabel="Repetition Ratio", xticks=x, xticklabels=models,
            ylim=(0, 1.0))
    ax2.tick_params(axis="x", rotation=20); ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rhythm_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[rhythm_score] Rhythm chart → {path}")


def plot_human_scores(results=None, out_dir=PLOTS_DIR):
    """Bar chart: human listening scores per model."""
    if results is None:
        results = PAPER_RESULTS
    models = list(results.keys())
    scores = [results[m]["human_score"] for m in models]
    colors = ["#c0c0c0", "#a0a0a0", "#4C72B0",
               "#5580B0", "#55A868", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, scores, color=colors[:len(models)],
                   alpha=0.85, edgecolor="black", lw=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=10)
    ax.axhline(3.0, color="orange", ls="--", alpha=0.5,
                label="Acceptable threshold (3.0)")
    ax.set(title="Human Listening Score Comparison (Table 3)",
           ylabel="Human Score [1–5]", ylim=(0, 6))
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "human_scores.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[rhythm_score] Human scores chart → {path}")


if __name__ == "__main__":
    plot_rhythm_comparison()
    plot_human_scores()