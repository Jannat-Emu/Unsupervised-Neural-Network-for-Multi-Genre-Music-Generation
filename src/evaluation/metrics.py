# src/evaluation/metrics.py
"""
Evaluation Metrics
===================
All metrics from the paper:

  1. Pitch Histogram Similarity  H(p,q) = Σ|p_i − q_i|
  2. Rhythm Diversity            D = #unique_durations / #notes
  3. Repetition Ratio            R = #repeated_4-step-patterns / #total
  4. Note Density                avg active pitches per time step
  5. Syncopation Index           fraction of off-beat onsets
  6. Reconstruction Loss         MSE (Tasks 1 & 2)
  7. Perplexity                  PPL = exp(L_TR / T)  (Task 3)
  8. Human Listening Score       survey [1–5]
"""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import N_PITCHES


# ── 1. Pitch histogram similarity ────────────────────────────────────

def pitch_histogram(roll: np.ndarray) -> np.ndarray:
    """12-bin pitch-class histogram from (T, 88) piano-roll."""
    act  = roll.sum(axis=0)   # (88,)
    hist = np.zeros(12)
    for i in range(act.shape[0]):
        hist[i % 12] += act[i]
    s = hist.sum()
    return hist / s if s > 0 else hist


def pitch_histogram_similarity(gen: np.ndarray,
                                 ref: np.ndarray) -> float:
    """H(p,q) = Σ_{i=1}^{12} |p_i − q_i|  ∈ [0, 2]."""
    return float(np.sum(np.abs(pitch_histogram(gen) - pitch_histogram(ref))))


# ── 2. Rhythm diversity ───────────────────────────────────────────────

def rhythm_diversity(roll: np.ndarray, threshold=0.1) -> float:
    """D = #unique_durations / #total_notes (higher → more rhythmic variety)."""
    durs = []
    for p in range(roll.shape[1]):
        col    = roll[:, p]
        dur, a = 0, False
        for v in col:
            if v >= threshold:
                a = True; dur += 1
            else:
                if a and dur > 0:
                    durs.append(dur)
                a = False; dur = 0
        if a and dur > 0:
            durs.append(dur)
    if not durs: return 0.0
    return len(set(durs)) / len(durs)


# ── 3. Repetition ratio ───────────────────────────────────────────────

def repetition_ratio(roll: np.ndarray, pattern_len=4) -> float:
    """R = #repeated 4-step patterns / #total patterns (lower → more creative)."""
    from collections import Counter
    binary   = (roll > 0.1).astype(np.uint8)
    T        = binary.shape[0]
    if T < pattern_len: return 0.0
    patterns = [tuple(binary[t:t+pattern_len].flatten())
                for t in range(T - pattern_len + 1)]
    counts   = Counter(patterns)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(patterns)


# ── 4. Note density ───────────────────────────────────────────────────

def note_density(roll: np.ndarray, threshold=0.1) -> float:
    """Average number of simultaneously active pitches per time step."""
    return float((roll > threshold).sum(axis=1).mean())


# ── 5. Syncopation index ──────────────────────────────────────────────

def syncopation_index(roll: np.ndarray, threshold=0.1,
                       steps_per_beat=4) -> float:
    """Fraction of note onsets that occur on off-beat steps."""
    onsets = 0; offbeat = 0
    for p in range(roll.shape[1]):
        prev = False
        for t, v in enumerate(roll[:, p]):
            on_now = v >= threshold
            if on_now and not prev:
                onsets += 1
                if t % steps_per_beat != 0:
                    offbeat += 1
            prev = on_now
    return offbeat / onsets if onsets > 0 else 0.0


# ── 6. Batch evaluation ───────────────────────────────────────────────

def evaluate_batch(rolls: np.ndarray,
                    ref_rolls: np.ndarray | None = None) -> dict:
    """Run all metrics on a batch of piano-rolls (N, T, 88)."""
    rds  = [rhythm_diversity(r)  for r in rolls]
    rrs  = [repetition_ratio(r)  for r in rolls]
    nds  = [note_density(r)      for r in rolls]
    syns = [syncopation_index(r) for r in rolls]

    result = {
        "rhythm_diversity"   : float(np.mean(rds)),
        "repetition_ratio"   : float(np.mean(rrs)),
        "note_density"       : float(np.mean(nds)),
        "syncopation_index"  : float(np.mean(syns)),
    }

    if ref_rolls is not None and len(ref_rolls) > 0:
        sims = [pitch_histogram_similarity(rolls[i % len(rolls)],
                                            ref_rolls[i % len(ref_rolls)])
                for i in range(min(len(rolls), 100))]
        result["pitch_hist_sim"] = float(np.mean(sims))

    return result


# ── 7. Print report ───────────────────────────────────────────────────

def print_metrics(name: str, m: dict):
    print(f"\n── {name} ──")
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<26} {v:.4f}")
        else:
            print(f"  {k:<26} {v}")


if __name__ == "__main__":
    # Demo: random roll
    roll = np.random.rand(64, 88).astype(np.float32) * 0.15
    m    = evaluate_batch(roll[None])
    print_metrics("Random baseline demo", m)