import numpy as np
import torch
from torch.utils.data import DataLoader
from unsupervised import DSpritesLazyDataset, LatentEncoder

# === Load dataset ===
dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

# === Load trained encoder ===
encoder = LatentEncoder(z_dim=10)  # adjust z_dim if needed
encoder.load_state_dict(torch.load('./unsupervised_checkpoints/encoder_epoch_20.pt', map_location='cpu'))
encoder.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = encoder.to(device)

# === Encode latents ===
all_z = []
all_factors = []

with torch.no_grad():
    for i, (x, latents) in enumerate(dataloader):
        if len(all_z) >= 10000:  # Google library defaults to 10k
            break
        x = x.to(device)
        z = encoder(x).cpu().numpy()  # (B, z_dim)
        y = latents[:, 1:6].numpy()   # ground-truth factors

        all_z.append(z)
        all_factors.append(y)

# === Save as NPZ ===
z = np.concatenate(all_z, axis=0)
factors = np.concatenate(all_factors, axis=0)
np.savez('representation.npz', representations=z, factors=factors)

print(f"Saved to 'representation.npz' with shape: z={z.shape}, factors={factors.shape}")
