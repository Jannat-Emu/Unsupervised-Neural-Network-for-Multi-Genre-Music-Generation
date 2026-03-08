# src/training_task/train_vae_task2.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

from src.models.vae import MusicVAE


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/processed/tokens"
CHECKPOINT_PATH = "checkpoints/checkpoint_vae.pt"


# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001


# -----------------------------
# Load dataset
# -----------------------------
print("Loading token dataset...")

data = np.load(os.path.join(DATA_PATH, "tokens_all.npy"))

data_tensor = torch.tensor(data, dtype=torch.float32)

# reshape to (batch, seq_len, features)
data_tensor = data_tensor.unsqueeze(1)

dataset = TensorDataset(data_tensor)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# -----------------------------
# Model
# -----------------------------
model = MusicVAE(input_dim=64).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)


# -----------------------------
# Training loop
# -----------------------------
print("Training VAE...")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total_recon = 0
    total_kl = 0

    for batch in loader:

        x = batch[0].to(device)

        optimizer.zero_grad()

        x_hat, mu, log_var = model(x)

        loss, recon, kl = model.loss(x, x_hat, mu, log_var)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    avg_loss = total_loss / len(loader)
    avg_recon = total_recon / len(loader)
    avg_kl = total_kl / len(loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"Recon: {avg_recon:.4f} | "
        f"KL: {avg_kl:.4f}"
    )


# -----------------------------
# Save checkpoint
# -----------------------------
os.makedirs("checkpoints", exist_ok=True)

torch.save(
    {
        "model_state": model.state_dict(),
        "input_dim": 64
    },
    CHECKPOINT_PATH
)

print("VAE training finished!")
print("Checkpoint saved to:", CHECKPOINT_PATH)