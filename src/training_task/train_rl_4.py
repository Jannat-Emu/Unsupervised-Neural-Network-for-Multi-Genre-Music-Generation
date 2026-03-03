# src/training/train_rl.py
"""
Task 4 Training – RLHF Policy Gradient
========================================
Algorithm 4 (from paper):
  for iteration in K:
    X_gen ~ p_θ(X)
    r = RewardModel(X_gen)   or   HumanScore(X_gen)
    ∇_θ J = E[r · ∇_θ log p_θ(X)]
    θ ← θ + η∇_θ J

Outputs:
  checkpoints/checkpoint_rlhf.pt
  outputs/plots/task4_rlhf.png
  outputs/generated_midis/task4_rl_sample_*.mid  (10 samples)
  outputs/survey_results/rl_comparison.json
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (DEVICE, RL_STEPS, RL_LR, WEIGHT_DECAY,
                         GRAD_CLIP, CKPT_TRANS, CKPT_RLHF,
                         PLOTS_DIR, SURVEY_DIR, N_GEN_RL,
                         VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
                         PITCH_OFFSET, VEL_OFFSET, DUR_OFFSET, N_VEL_BINS)
from src.models.transformer import MusicTransformer
from src.generation.generate_music import tokens_to_roll
from src.generation.midi_export import pianoroll_to_midi


# ── Rule-based reward ─────────────────────────────────────────────────

def rule_reward(toks: list) -> float:
    """
    Fast heuristic reward in [0, 1].
    Combines: pitch diversity, velocity range, non-repetition, length.
    """
    from collections import Counter
    pitches = [t - PITCH_OFFSET for t in toks
               if PITCH_OFFSET <= t < VEL_OFFSET]
    vels    = [t - VEL_OFFSET   for t in toks
               if VEL_OFFSET   <= t < DUR_OFFSET]
    if not pitches:
        return 0.0
    diversity   = min(len(set(pitches)) / 12.0, 1.0)
    vel_range   = (max(vels) - min(vels)) / 32.0 if vels else 0.0
    counts      = Counter(toks)
    rep         = 1.0 - min(max(counts.values()) / len(toks), 0.8)
    len_score   = min(len(toks) / 100.0, 1.0)
    return float(np.clip(
        0.4 * diversity + 0.2 * vel_range + 0.3 * rep + 0.1 * len_score,
        0.0, 1.0))


# ── Policy gradient step ──────────────────────────────────────────────

def pg_step(model, opt, n_samples=8, max_len=128,
            device="cpu", human_scores=None):
    """
    One REINFORCE step.  Returns mean reward.
    ∇_θ J = E[(r − baseline) · ∇_θ log p_θ(X)]
    """
    model.train()
    log_probs_list, tok_seqs = [], []

    for _ in range(n_samples):
        tokens   = torch.tensor([[BOS_TOKEN]], device=device)
        lp_accum = []
        for _ in range(max_len - 1):
            logits = model(tokens)[:, -1, :] / 1.0
            probs  = torch.softmax(logits, -1)
            nxt    = torch.multinomial(probs, 1).item()
            lp_accum.append(torch.log(probs[0, nxt] + 1e-9))
            tokens = torch.cat([tokens,
                                  torch.tensor([[nxt]], device=device)], dim=1)
            if nxt in (PAD_TOKEN, EOS_TOKEN):
                break
        log_probs_list.append(torch.stack(lp_accum).sum())
        tok_seqs.append(tokens[0].tolist())

    # Compute rewards
    if human_scores:
        avg_h = np.mean(list(human_scores.values()))
        r_list = [(avg_h - 1.0) / 4.0] * n_samples
    else:
        r_list = [rule_reward(s) for s in tok_seqs]

    rewards   = torch.tensor(r_list, dtype=torch.float32, device=device)
    baseline  = rewards.mean()
    adv       = rewards - baseline
    lp_tensor = torch.stack(log_probs_list)
    pg_loss   = -(adv.detach() * lp_tensor).mean()

    opt.zero_grad(); pg_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step()
    return float(rewards.mean())


# ── Main training ─────────────────────────────────────────────────────

def train(args):
    device = args.device
    model  = MusicTransformer().to(device)

    # Load pre-trained Transformer if available
    if os.path.exists(CKPT_TRANS):
        ckpt = torch.load(CKPT_TRANS, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Task 4] Loaded pre-trained Transformer from {CKPT_TRANS}")
    else:
        print("[Task 4] No pre-trained checkpoint found — using random init.")

    opt = torch.optim.Adam(model.parameters(), lr=args.rl_lr,
                            weight_decay=WEIGHT_DECAY)

    # Load human survey if present
    human_scores = None
    hf = os.path.join(SURVEY_DIR, "human_scores.json")
    if os.path.exists(hf):
        with open(hf) as f:
            human_scores = json.load(f)
        print(f"[Task 4] Loaded {len(human_scores)} human scores.")

    # Pre-RL baseline
    model.eval()
    pre_r = []
    for i in range(5):
        toks = model.generate(max_len=256, device=device)
        pre_r.append(rule_reward(toks))
        roll = tokens_to_roll(toks)
        pianoroll_to_midi(roll,
            os.path.join(args.out_dir, f"task4_before_rl_{i+1:02d}.mid"))
    print(f"[Task 4] Pre-RL avg reward: {np.mean(pre_r):.4f}")

    # RL training
    reward_hist = []
    print(f"[Task 4] RLHF: {args.rl_steps} steps ...")
    for step in range(1, args.rl_steps + 1):
        r = pg_step(model, opt, n_samples=8, max_len=128,
                    device=device, human_scores=human_scores)
        reward_hist.append(r)
        if step % 20 == 0 or step == 1:
            print(f"  Step {step:04d}/{args.rl_steps}  AvgReward={r:.4f}")

    # Post-RL generation
    model.eval()
    post_r = []
    for i in range(N_GEN_RL):
        toks = model.generate(max_len=512, temperature=0.9, device=device)
        post_r.append(rule_reward(toks))
        roll = tokens_to_roll(toks)
        pianoroll_to_midi(roll,
            os.path.join(args.out_dir, f"task4_rl_sample_{i+1:02d}.mid"))
        print(f"  Generated: task4_rl_sample_{i+1:02d}.mid")

    # Save
    os.makedirs(os.path.dirname(CKPT_RLHF), exist_ok=True)
    torch.save({"model_state"   : model.state_dict(),
                "reward_history": reward_hist,
                "pre_rewards"   : pre_r,
                "post_rewards"  : post_r}, CKPT_RLHF)
    print(f"[Task 4] Checkpoint → {CKPT_RLHF}")

    _plot(reward_hist, pre_r, post_r)

    comparison = {
        "pre_rl_reward" : float(np.mean(pre_r)),
        "post_rl_reward": float(np.mean(post_r)),
        "improvement"   : float(np.mean(post_r) - np.mean(pre_r)),
    }
    os.makedirs(SURVEY_DIR, exist_ok=True)
    with open(os.path.join(SURVEY_DIR, "rl_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 45)
    print(f"  Pre-RL  avg reward : {comparison['pre_rl_reward']:.4f}")
    print(f"  Post-RL avg reward : {comparison['post_rl_reward']:.4f}")
    print(f"  Improvement        : {comparison['improvement']:+.4f}")
    print("=" * 45)
    return model


def _plot(hist, pre, post):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist, color="#4C72B0")
    ax1.axhline(np.mean(pre), color="red", ls="--",
                label=f"Pre-RL ({np.mean(pre):.3f})")
    ax1.set(title="Task 4 – RLHF Reward", xlabel="RL Step", ylabel="Reward")
    ax1.legend(); ax1.grid(alpha=0.3)

    bars = ax2.bar(["Pre-RL", "Post-RL"], [np.mean(pre), np.mean(post)],
                    color=["#DD8452", "#55A868"], alpha=0.85, width=0.5)
    ax2.bar_label(bars, fmt="%.3f", padding=3)
    ax2.set(title="Task 4 – Before vs After RLHF",
            ylabel="Avg Reward Score", ylim=(0, 1.1))
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "task4_rlhf.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[Task 4] Plot → {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rl_steps", type=int,   default=RL_STEPS)
    p.add_argument("--rl_lr",    type=float, default=RL_LR)
    p.add_argument("--device",   type=str,   default=DEVICE)
    p.add_argument("--out_dir",  type=str,
                   default=os.path.join(os.path.dirname(__file__),
                                         "..", "..", "outputs", "generated_midis"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n{'='*55}\nTask 4 – RLHF  |  steps={args.rl_steps}  device={args.device}\n{'='*55}")
    train(args)
    print("✓ Task 4 complete.")