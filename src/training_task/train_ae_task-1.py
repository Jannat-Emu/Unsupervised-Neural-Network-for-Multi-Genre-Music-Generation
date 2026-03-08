import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim

from src.models.autoencoder import LSTMAutoencoder

# Paths
DATA_PATH = "data/processed/tokens"
CHECKPOINT_PATH = "checkpoints/checkpoint_ae.pt"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001
LATENT_DIM = 128

# Load token dataset
data = np.load(os.path.join(DATA_PATH, "tokens_all.npy"))

data_tensor = torch.tensor(data, dtype=torch.float32)
data_tensor = data_tensor.unsqueeze(1)

dataset = TensorDataset(data_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = LSTMAutoencoder(input_dim=data.shape[-1], latent_dim=LATENT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Training Autoencoder...")

for epoch in range(EPOCHS):

    total_loss = 0

    for batch in loader:

        x = batch[0].to(device)

        optimizer.zero_grad()

        x_hat, _ = model(x)

        loss = criterion(x_hat, x)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {avg_loss:.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)

torch.save({
    "model_state": model.state_dict(),
    "input_dim": 64
}, CHECKPOINT_PATH)

print("Autoencoder training finished!")