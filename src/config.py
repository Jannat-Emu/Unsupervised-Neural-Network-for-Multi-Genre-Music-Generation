import os
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Data paths ───────────────────────────────────────────────
DATA_DIR        = os.path.join(ROOT, "data")
RAW_MIDI_DIR    = os.path.join(DATA_DIR, "raw_midi")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
PIANO_ROLL_DIR  = os.path.join(PROCESSED_DIR, "piano_rolls")
TOKEN_DIR       = os.path.join(PROCESSED_DIR, "tokens")
META_DIR        = os.path.join(PROCESSED_DIR, "metadata")
SPLIT_DIR       = os.path.join(DATA_DIR, "train_test_split")

# ── Output paths ─────────────────────────────────────────────
OUTPUT_DIR      = os.path.join(ROOT, "outputs")
MIDI_OUT_DIR    = os.path.join(OUTPUT_DIR, "generated_midis")
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
SURVEY_DIR      = os.path.join(OUTPUT_DIR, "survey_results")

# ── Checkpoint paths ─────────────────────────────────────────
CKPT_DIR        = os.path.join(ROOT, "checkpoints")
CKPT_AE         = os.path.join(CKPT_DIR, "checkpoint_ae.pt")
CKPT_VAE        = os.path.join(CKPT_DIR, "checkpoint_vae.pt")
CKPT_TRANS      = os.path.join(CKPT_DIR, "checkpoint_transformer.pt")
CKPT_RLHF       = os.path.join(CKPT_DIR, "checkpoint_rlhf.pt")

# ── Report ───────────────────────────────────────────────────
REPORT_DIR      = os.path.join(ROOT, "report")

# ── MAESTRO dataset ──────────────────────────────────────────
MAESTRO_URL = (
    "https://storage.googleapis.com/magentadata/datasets/maestro"
    "/v3.0.0/maestro-v3.0.0-midi.zip"
)

# ── MIDI / Piano-roll parameters ─────────────────────────────
PITCH_MIN       = 21
PITCH_MAX       = 108
N_PITCHES       = PITCH_MAX - PITCH_MIN + 1
STEPS_PER_BEAT  = 4
STEPS_PER_BAR   = 16
SEQ_LEN         = 64
DEFAULT_TEMPO   = 120.0

# ── Dataset split ────────────────────────────────────────────
TEST_SPLIT  = 0.15
VAL_SPLIT   = 0.10

# ── Tokenizer vocabulary ─────────────────────────────────────
VOCAB_SIZE    = 139
PAD_TOKEN     = 0
BOS_TOKEN     = 1
EOS_TOKEN     = 2
PITCH_OFFSET  = 3
VEL_OFFSET    = 91
DUR_OFFSET    = 123
N_VEL_BINS    = 32
N_DUR_BINS    = 16

# ── Training hyper-parameters ────────────────────────────────
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── Task 1 - LSTM Autoencoder ────────────────────────────────
AE_HIDDEN_DIM  = 256
AE_LATENT_DIM  = 64
AE_NUM_LAYERS  = 2
AE_DROPOUT     = 0.3
N_GEN_AE       = 5

# ── Task 2 - VAE ─────────────────────────────────────────────
VAE_HIDDEN_DIM  = 256
VAE_LATENT_DIM  = 128
VAE_NUM_LAYERS  = 2
VAE_BETA        = 0.5
N_GEN_VAE       = 8

# ── Task 3 - Transformer ─────────────────────────────────────
TRANS_D_MODEL    = 256
TRANS_NHEAD      = 8
TRANS_NUM_LAYERS = 6
TRANS_DIM_FF     = 1024
TRANS_DROPOUT    = 0.1
TRANS_MAX_LEN    = 512
N_GEN_TRANS      = 10

# ── Task 4 - RLHF ────────────────────────────────────────────
RL_STEPS   = 50     # was 200
RL_LR      = 1e-4
N_GEN_RL   = 5      # was 10

# ── Auto-create all directories on import ────────────────────
for _dir in [RAW_MIDI_DIR, PIANO_ROLL_DIR, TOKEN_DIR, META_DIR,
             SPLIT_DIR, MIDI_OUT_DIR, PLOTS_DIR, SURVEY_DIR, CKPT_DIR]:
    os.makedirs(_dir, exist_ok=True)