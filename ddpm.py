import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from tqdm import tqdm
import torch.optim as optim




class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path):
        # Lazily load the .npz file to avoid memory overhead
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx], dtype=np.float32)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img * 2.0 - 1.0
        return img

    @classmethod
    def make_loader(cls, npz_path, batch_size=128, num_workers=8):
        dataset = cls(npz_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )



# Sinusoidal timestep embedding used in transformer-style positional encoding
def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb  # [B, dim]




# A basic 2D convolution block with optional timestep embedding
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.conv1(x))
        h += self.time_proj(t_emb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        return h + self.skip(x)




# The UNet backbone for the DDPM model
class DDPM(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.down1 = ResidualBlock(img_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up3 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        self.up1 = ResidualBlock(base_channels, base_channels, time_emb_dim)

        self.out = nn.Conv2d(base_channels, img_channels, 1)

        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)

        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        m = self.mid(self.pool(d3), t_emb)

        u3 = self.up3(self.upsample(m) + d3, t_emb)
        u2 = self.up2(self.upsample(u3) + d2, t_emb)
        u1 = self.up1(self.upsample(u2) + d1, t_emb)

        return self.out(u1)





class SampleGenerator:
    def __init__(self, model, img_size=64, save_dir="samples", num_steps=1000, device='cuda'):
        """
        Args:
            model: Trained DDPM model
            img_size: Image resolution (assumes square)
            save_dir: Directory to save generated images
            num_steps: Number of diffusion steps (T)
            device: CUDA or CPU
        """
        self.model = model.to(device)
        self.device = device
        self.img_size = img_size
        self.save_dir = save_dir
        self.num_steps = num_steps
        os.makedirs(save_dir, exist_ok=True)

        # Precompute betas and related coefficients
        self.betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)


    def sample(self, batch_size=64):
        """
        Generate a batch of images from pure noise using reverse diffusion.
        """
        x = torch.randn(batch_size, 1, self.img_size, self.img_size, device=self.device)  # start from pure noise

        for t in tqdm(reversed(range(self.num_steps)), desc="Sampling", leave=False):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            with torch.no_grad():
                predicted_noise = self.model(x, t_tensor)

            alpha_t = self.alphas[t]
            alpha_hat_t = self.alpha_hats[t]
            beta_t = self.betas[t]

            # Correct posterior mean from DDPM
            mean = (1 / alpha_t.sqrt()) * (
                x - ((1 - alpha_t) / (1 - alpha_hat_t).sqrt()) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean

        x = x.clamp(-1, 1)
        x = (x + 1) / 2  # scale to [0, 1]
        return x



    def save_samples(self, epoch, batch_size=64, grid_rows=8):
        """
        Generate and save sample images to disk.
        """
        samples = self.sample(batch_size)
        grid = vutils.make_grid(samples, nrow=grid_rows, normalize=True)
        save_path = os.path.join(self.save_dir, f"epoch_{epoch:04d}.png")
        vutils.save_image(grid, save_path)
        print(f"[✓] Saved samples to {save_path}")





def train_ddpm(
    npz_path='dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
    epochs=100,
    batch_size=128,
    lr=2e-4,
    img_size=64,
    save_dir="checkpoints",
    sample_dir="samples",
    resume=True,
    num_steps=1000,
    device="cuda"
):
    # Setup directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Model, optimizer, loss
    model = DDPM(img_channels=1, base_channels=64, time_emb_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Noise schedule
    betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alpha_hats = torch.cumprod(alphas, dim=0)

    # Resume from checkpoint if available
    start_epoch = 0
    checkpoint_path = os.path.join(save_dir, "latest.pt")
    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[✓] Resumed from checkpoint at epoch {start_epoch}")

    # Data loader
    loader = DSpritesLazyDataset.make_loader(npz_path, batch_size=batch_size)

    # Sampler for saving generated images
    sampler = SampleGenerator(model, img_size=img_size, save_dir=sample_dir, num_steps=num_steps, device=device)

    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for x in tqdm(loader, desc=f"Epoch {epoch}"):
            x = x.to(device)
            t = torch.randint(0, num_steps, (x.shape[0],), device=device).long()
            noise = torch.randn_like(x)
            alpha_hat = alpha_hats[t][:, None, None, None]
            x_t = alpha_hat.sqrt() * x + (1 - alpha_hat).sqrt() * noise
            noise_pred = model(x_t, t)
            loss = mse(noise_pred, noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if total_batches % 100 == 0:
                tqdm.write(f"[step {total_batches:5d}] timestep {t[0].item():4d}: loss = {loss.item():.6f}")
            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        print(f"[✓] Epoch {epoch} avg loss: {avg_loss:.6f}")
        sampler.save_samples(epoch)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, checkpoint_path)
        print(f"[✓] Checkpoint saved for epoch {epoch}")




if __name__ == "__main__":
    train_ddpm(
        npz_path="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", 
        epochs=100,
        batch_size=128,
        lr=2e-4,
        img_size=64,
        save_dir="checkpoints",
        sample_dir="samples",
        resume=True,
        num_steps=1000,
        device="cuda"
    )


