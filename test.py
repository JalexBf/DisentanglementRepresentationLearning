import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import os
from SimpleDiff import UNet, DDPM  

# Load model + DDPM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(base_channels=32).to(device)
ddpm = DDPM(model)

checkpoint = torch.load("./simple_checkpoints/ddpm_epoch_10.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Traversal function
def run_disentanglement_traversals(ddpm, base_latent, save_path="traversals.png"):
    ddpm.model.eval()
    factor_dims = {0: 3, 1: 6, 2: 40, 3: 32, 4: 32}

    all_samples = []

    for dim, max_val in factor_dims.items():
        values = torch.linspace(0, max_val - 1, steps=10).long()
        latents = torch.stack([
            base_latent.clone().index_fill_(0, torch.tensor([dim]), v) for v in values
        ]).to(device)

        with torch.no_grad():
            samples = ddpm.sample(img_size=64, y=latents, batch_size=len(latents))
            samples = (samples + 1) * 0.5
        all_samples.append(samples)

    grid = make_grid(torch.cat(all_samples, dim=0), nrow=10)
    os.makedirs("traversals", exist_ok=True)
    save_image(grid, f"traversals/{save_path}")
    print(f"Saved: traversals/{save_path}")

# Run
base_latent = torch.tensor([0, 3, 20, 16, 16], dtype=torch.long).to(device)
run_disentanglement_traversals(ddpm, base_latent)
