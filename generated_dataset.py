import torch
import numpy as np
import os
from SimpleDiff import DDPM, UNet

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_size = 64
num_samples = 10000
batch_size = 128
save_dir = './generated_samples'
os.makedirs(save_dir, exist_ok=True)

# Load the model checkpoint
checkpoint_path = './checkpoints/ddpm_epoch_20.pt'
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Initialize model and DDPM
model = UNet(base_channels=32).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

ddpm = DDPM(model)

# Latent space configurations based on dSprites
latent_factors = {
    'color': [0],  # Always 0 since dSprites is binary
    'shape': list(range(3)),
    'scale': list(range(6)),
    'orient': list(range(40)),
    'posX': list(range(32)),
    'posY': list(range(32))
}

# Generate samples and latents
generated_imgs = []
generated_latents = []

for batch_start in range(0, num_samples, batch_size):
    batch_end = min(batch_start + batch_size, num_samples)
    batch_size_actual = batch_end - batch_start

    # Prepare latents for the batch
    batch_latents = []
    for _ in range(batch_size_actual):
        shape = np.random.choice(latent_factors['shape'])
        scale = np.random.choice(latent_factors['scale'])
        orient = np.random.choice(latent_factors['orient'])
        posX = np.random.choice(latent_factors['posX'])
        posY = np.random.choice(latent_factors['posY'])
        latent = [0, shape, scale, orient, posX, posY]
        batch_latents.append(latent)

    # Convert to tensor
    batch_latents = torch.tensor(batch_latents, dtype=torch.long, device=device)

    print(f"Generating batch {batch_start // batch_size + 1} / {num_samples // batch_size + 1}...")

    # Generate samples
    with torch.no_grad():
        generated_batch = ddpm.sample(img_size, batch_latents)
        generated_batch = (generated_batch + 1) * 0.5  # Scale to [0, 1]
        generated_batch = generated_batch.cpu().numpy()  # Shape: (B, 1, 64, 64)

    # Collect generated images and latents as batches
    generated_imgs.append(generated_batch)
    generated_latents.append(batch_latents.cpu().numpy())

# Concatenate all batches
generated_imgs = np.concatenate(generated_imgs, axis=0).astype(np.float32)  # Shape: (N, 1, 64, 64)
generated_latents = np.concatenate(generated_latents, axis=0).astype(np.int32)  # Shape: (N, 6)

# Save as NPZ
np.savez_compressed(
    os.path.join(save_dir, 'generated_dsprites.npz'),
    imgs=generated_imgs,
    latents_classes=generated_latents
)

print(f"Generated dataset saved at {os.path.join(save_dir, 'generated_dsprites.npz')}")
