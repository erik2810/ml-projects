import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
DATA_DIR = BASE_DIR / "data"

CHECKPOINT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model defaults — kept here so the API and training scripts stay in sync
GNN_DEFAULTS = {
    "hidden_dim": 32,
    "num_layers": 3,
    "dropout": 0.3,
    "lr": 0.005,
    "epochs": 200,
}

GENERATOR_DEFAULTS = {
    "max_nodes": 20,
    "latent_dim": 32,
    "hidden_dim": 256,
    "lr": 1e-3,
    "epochs": 300,
    "num_training_graphs": 2000,
}

VAE_DEFAULTS = {
    "max_nodes": 20,
    "latent_dim": 64,
    "hidden_dim": 128,
    "lr": 1e-3,
    "epochs": 200,
}

DIFFUSION_DEFAULTS = {
    "max_nodes": 20,
    "hidden_dim": 64,
    "timesteps": 50,
    "lr": 1e-3,
    "epochs": 150,
}

SPATIAL_VAE_DEFAULTS = {
    "max_nodes": 64,
    "latent_dim": 32,
    "hidden_dim": 64,
    "lr": 1e-3,
    "epochs": 100,
}

SPATIAL_DIFFUSION_DEFAULTS = {
    "max_nodes": 64,
    "hidden_dim": 64,
    "timesteps": 50,
    "lr": 3e-4,
    "epochs": 100,
}
