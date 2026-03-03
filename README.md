<!-- # Unsupervised Neural Network Music Generation
**Course:** CSE425 / EEE474  ·  Spring 2026  ·  Deadline: 10 April 2026

---

## Project Structure

```
music-generation-unsupervised/
├── download_dataset.py         ← STEP 1: run this first
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw_midi/               ← MAESTRO .mid files land here
│   ├── processed/
│   │   ├── piano_rolls/        ← pianorolls.npy  (N, 64, 88)
│   │   ├── tokens/             ← tokens_all.npy  (N, 64)
│   │   └── metadata/
│   └── train_test_split/       ← train.txt / val.txt / test.txt
│
├── src/
│   ├── config.py               ← all paths & hyper-parameters
│   ├── preprocessing/
│   │   ├── midi_parser.py      ← MIDI → note events + splits
│   │   ├── piano_roll.py       ← MIDI → (T,88) arrays
│   │   └── tokenizer.py        ← events → integer tokens
│   ├── models/
│   │   ├── autoencoder.py      ← Task 1: LSTM AE
│   │   ├── vae.py              ← Task 2: β-VAE
│   │   ├── transformer.py      ← Task 3: Decoder-only Transformer
│   │   └── diffusion.py        ← Optional extension
│   ├── training/
│   │   ├── train_ae.py         ← Train Task 1
│   │   ├── train_vae.py        ← Train Task 2
│   │   ├── train_transformer.py← Train Task 3
│   │   └── train_rl.py         ← Train Task 4 (RLHF)
│   ├── evaluation/
│   │   ├── metrics.py          ← All metrics (Table 3)
│   │   ├── pitch_histogram.py  ← Pitch analysis plots
│   │   └── rhythm_score.py     ← Rhythm charts + Table 3 plot
│   └── generation/
│       ├── generate_music.py   ← CLI: generate from any model
│       ├── midi_export.py      ← piano-roll / events → .mid
│       └── sample_latent.py    ← VAE latent space explorer
│
├── notebooks/
│   ├── preprocessing.ipynb
│   └── baseline_markov.ipynb
│
├── outputs/
│   ├── generated_midis/        ← all .mid outputs
│   ├── plots/                  ← all .png charts
│   └── survey_results/         ← rl_comparison.json + human_scores.json
│
└── report/
    ├── final_report.tex
    ├── references.bib
    └── architecture_diagrams/
```

---

## Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download MAESTRO dataset
```bash
python download_dataset.py
```
This downloads **MAESTRO v3.0.0** (1 276 classical piano MIDI files, ~57 MB)
from Google Magenta and extracts all `.mid` files to `data/raw_midi/`.

If automatic download fails (network issues), follow the manual instructions:
```bash
python download_dataset.py --info
```

### Step 3 — Preprocess
```bash
python src/preprocessing/midi_parser.py      # creates train/val/test splits
python src/preprocessing/piano_roll.py       # builds (N, 64, 88) array
python src/preprocessing/tokenizer.py        # builds token sequences
```

### Step 4 — Train models
```bash
# Task 1 – LSTM Autoencoder (Easy)
python src/training/train_ae.py --epochs 50

# Task 2 – VAE (Medium)
python src/training/train_vae.py --epochs 50 --beta 0.5

# Task 3 – Transformer (Hard)
python src/training/train_transformer.py --epochs 50

# Task 4 – RLHF (Advanced)
python src/training/train_rl.py --rl_steps 200
```

### Step 5 — Evaluate & visualise
```bash
python src/evaluation/metrics.py
python src/evaluation/rhythm_score.py
python src/evaluation/pitch_histogram.py
```

### Step 6 — Generate music
```bash
python src/generation/generate_music.py --model ae    --n 5
python src/generation/generate_music.py --model vae   --n 8
python src/generation/generate_music.py --model trans --n 10 --temperature 0.9
python src/generation/generate_music.py --model rlhf  --n 10
```

---

## Dataset — MAESTRO v3.0.0

| Property | Value |
|----------|-------|
| Source | Google Magenta |
| URL | https://magenta.tensorflow.org/datasets/maestro |
| Genre | Classical Piano |
| Tracks | 1 276 |
| Duration | ~199 hours |
| Format | MIDI |
| License | CC BY-NC-SA 4.0 |
| Download size | ~57 MB (MIDI only) |

---

## Model Architectures

### Task 1 – LSTM Autoencoder
```
Encoder: BiLSTM(88→256) → Linear → z ∈ ℝ⁶⁴
Decoder: LSTM(64→256) → Linear → x̂ ∈ ℝ⁸⁸
Loss:    L_AE = ‖X − X̂‖²
```

### Task 2 – β-VAE
```
Encoder: BiLSTM → μ(X), log σ²(X)
Sample:  z = μ + σ ⊙ ε,  ε ~ N(0,I)
Loss:    L_VAE = ‖X − X̂‖² + β · D_KL
```

### Task 3 – Transformer
```
h_t = Emb(x_t) + PosEnc(t)
p(X) = ∏_t p(x_t | x_{<t})
Loss: L_TR = −Σ log p   PPL = exp(L_TR/T)
```

### Task 4 – RLHF
```
∇_θ J(θ) = E[r · ∇_θ log p_θ(X)]
r = rule-based reward or human survey score
```

---

## Expected Results (Table 3)

| Model | Loss | PPL | Rhythm Div ↑ | Human Score ↑ |
|-------|------|-----|-------------|---------------|
| Random | — | — | 0.12 | 1.1 |
| Markov | — | — | 0.31 | 2.3 |
| Task 1 AE | 0.82 | — | 0.45 | 3.1 |
| Task 2 VAE | 0.65 | — | 0.58 | 3.8 |
| Task 3 Trans | — | 12.5 | 0.72 | 4.4 |
| Task 4 RLHF | — | 11.2 | 0.79 | 4.8 |

---

## Human Survey (Task 4)
1. Generate survey samples: `python src/training/train_rl.py --survey_only`
2. Distribute MIDIs to ≥10 participants, ask them to rate 1–5
3. Save ratings to `outputs/survey_results/human_scores.json`:
   ```json
   {"survey_sample_01.mid": 4.0, "survey_sample_02.mid": 3.5, ...}
   ```
4. Re-run RLHF: `python src/training/train_rl.py` -->