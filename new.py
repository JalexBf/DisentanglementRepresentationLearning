import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
import math
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler


torch.manual_seed(0)
np.random.seed(0)


# ------------------------------
# Dataset Loader
# ------------------------------
class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path, cond_factors=('shape',)):
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']
        self.classes = self.data['latents_classes']  # integers [shape, scale, orientation, posX, posY]

        self.factor_names = ['color', 'shape', 'scale', 'orientation', 'posX', 'posY']
        self.factor_sizes = [1, 3, 6, 40, 32, 32]
        self.factor_indices = [self.factor_names.index(f) for f in cond_factors]
        self.cond_dim = sum([self.factor_sizes[i] for i in self.factor_indices])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)
        img = img * 2.0 - 1.0  # [-1, 1]

        factors = self.classes[idx]  # (6,)
        cond_parts = []
        for i in self.factor_indices:
            one_hot = torch.zeros(self.factor_sizes[i])
            one_hot[int(factors[i])] = 1.0
            cond_parts.append(one_hot)

        cond = torch.cat(cond_parts)  # final one-hot vector

        return img, cond




# ------------------------------
# Sinusoidal Positional Embedding
# ------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# ------------------------------
# DDPM U-Net Model
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.shortcut(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, cond_dim=0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.cond_proj = nn.Linear(cond_dim, time_emb_dim) if cond_dim > 0 else None

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.conv_in = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.down1 = ResidualBlock(c1, c2, time_emb_dim)
        self.downsample1 = Downsample(c2)

        self.down2 = ResidualBlock(c2, c3, time_emb_dim)
        self.downsample2 = Downsample(c3)

        self.bot1 = ResidualBlock(c3, c3, time_emb_dim)
        self.bot2 = ResidualBlock(c3, c3, time_emb_dim)

        self.up1 = Upsample(c3)
        self.up_block1 = ResidualBlock(c2 + c2, c2, time_emb_dim)  # c3 -> up1 -> c2; skip = c2

        self.up2 = Upsample(c2)
        self.up_block2 = ResidualBlock(c1 + c1, c1, time_emb_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, 3, padding=1)
        )


    def forward(self, x, t, cond=None):
        t_emb = self.time_mlp(t)
        if self.cond_proj is not None and cond is not None:
            t_emb = t_emb + self.cond_proj(cond)

        x1 = self.conv_in(x)                   # (batch, base, 64, 64)
        x2 = self.down1(x1, t_emb)              # (batch, base*2, 64, 64)
        x2_down = self.downsample1(x2)          # (batch, base*2, 32, 32)
        x3 = self.down2(x2_down, t_emb)         # (batch, base*4, 32, 32)
        x3_down = self.downsample2(x3)          # (batch, base*4, 16, 16)

        x4 = self.bot1(x3_down, t_emb)           # (batch, base*4, 16, 16)
        x4 = self.bot2(x4, t_emb)

        x = self.up1(x4)                        # (batch, base*2, 32, 32)
        x = torch.cat((x, x2_down), dim=1)       # (batch, base*2 + base*2, 32, 32)
        x = self.up_block1(x, t_emb)

        x = self.up2(x)                         # (batch, base, 64, 64)
        x = torch.cat((x, x1), dim=1)            # (batch, base + base, 64, 64)
        x = self.up_block2(x, t_emb)

        return self.conv_out(x)




# ------------------------------
# DDPM Core
# ------------------------------
class DDPM:
    def __init__(self, model, timesteps=500, beta_start=1e-4, beta_end=0.01):
        self.model = model
        device = next(model.parameters()).device
        self.T = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_schedule(self, x0, t):
        noise = torch.randn_like(x0)
        batch_size = x0.size(0)
        sqrt_alpha_hat = self.alpha_hat[t].reshape(batch_size, 1, 1, 1) ** 0.5
        sqrt_one_minus = (1 - self.alpha_hat[t]).reshape(batch_size, 1, 1, 1) ** 0.5
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus * noise
        return xt, noise

    def train_step(self, x0, cond):
        self.model.train()
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)
        xt, noise = self.noise_schedule(x0, t)
        pred_noise = self.model(xt, t, cond)
        return F.mse_loss(pred_noise, noise)


    @torch.no_grad()
    def sample(self, img_size, batch_size=256):
        device = next(self.model.parameters()).device
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)
        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.model(img, time)

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            if t > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise

        return img.clamp(-1, 1)



# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', cond_factors=('shape',))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    # Build and compile once
    model = UNet(base_channels=64, cond_dim=3).to(device)  # if using 'shape' (3 classes)
    model = torch.compile(model)

    # Create EMA once
    ema_model = deepcopy(model)
    ema_decay = 0.995

    ddpm = DDPM(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    os.makedirs('./new_checkpoints', exist_ok=True)
    os.makedirs('./new_samples', exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    save_every = 1

    fixed_noise = torch.randn(16, 1, 64, 64, device=device)

    start_epoch = 1  # default starting point

    checkpoint_dir = './new_checkpoints'
    latest_ckpt = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )


    checkpoint = None

    if latest_ckpt:
        path = os.path.join(checkpoint_dir, latest_ckpt[-1])
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    if checkpoint is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        print("No EMA state found in checkpoint. Reinitializing EMA.")




    for epoch in range(start_epoch, 100):
        for i, (batch, cond) in enumerate(dataloader):
            batch = batch.to(device)
            cond = cond.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = ddpm.train_step(batch, cond)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if i % 100 == 0:
                with torch.no_grad():
                    t = torch.randint(0, ddpm.T, (batch.size(0),), device=batch.device)
                    xt, noise = ddpm.noise_schedule(batch, t)
                    mean_xt = xt.abs().mean().item()
                    mean_noise = noise.abs().mean().item()
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | Mean xt abs: {mean_xt:.4f} | Mean noise abs: {mean_noise:.4f}")

        # Save model and EMA samples
        if epoch % save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, f'./new_checkpoints/ddpm_epoch_{epoch}.pt')


            print(f"Saved checkpoint at epoch {epoch}")

            # Use EMA model temporarily for sampling
            ddpm.model = ema_model
            with torch.no_grad():
                img = fixed_noise.clone()
                for t in reversed(range(ddpm.T)):
                    time = torch.full((img.size(0),), t, device=device, dtype=torch.long)
                    cond = torch.zeros(img.size(0), 3, device=device)  # default to class 0 (square)
                    pred_noise = ddpm.model(img, time, cond)
                    alpha = ddpm.alpha[t]
                    alpha_hat = ddpm.alpha_hat[t]
                    beta = ddpm.beta[t]
                    noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
                    img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise
                sampled_imgs = img.clamp(-1, 1)

            sampled_imgs = (sampled_imgs + 1) * 0.5
            save_image(sampled_imgs, f'./new_samples/sample_epoch_{epoch}.png', nrow=4)
            print(f"Saved samples at epoch {epoch}")

            # === Sample Quality Metrics ===
            with torch.no_grad():
                # Use a small fixed batch of real samples
                real_batch = next(iter(dataloader))[:16].to(device)
                gen_batch = sampled_imgs[:16].to(device)

                # L1 Distance (in [-1, 1] space)
                l1 = F.l1_loss(gen_batch * 2 - 1, real_batch).item()

                # Binariness score
                binariness = ((gen_batch < 0.05) | (gen_batch > 0.95)).float().mean().item()

                print(f"Sample Quality @ Epoch {epoch} | L1: {l1:.4f} | Binariness: {binariness:.4f}")

                # Save to text file
                with open("./new_samples/sample_quality.txt", "a") as f:
                    f.write(f"{epoch},{l1:.4f},{binariness:.4f}\n")

                pure_black = (sampled_imgs.view(sampled_imgs.size(0), -1).max(dim=1).values < 0.05).float().mean().item()
                print(f"Pure black ratio: {pure_black:.4f}")


            # Switch back to main model
            ddpm.model = model



        scheduler.step()
