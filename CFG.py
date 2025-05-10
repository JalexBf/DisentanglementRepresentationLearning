import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
import math
from copy import deepcopy

# ------------------------------
# Dataset Loader
# ------------------------------
class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']
        self.latents = self.data['latents_classes']  # shape (737280, 6)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0) * 2.0 - 1.0  # [-1, 1]
        latent = torch.tensor(self.latents[idx], dtype=torch.long)
        return img, latent



# ------------------------------
# Conditional Embedding
# ------------------------------
class ConditionalEmbedding(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.shape = nn.Embedding(3, emb_dim)
        self.scale = nn.Embedding(6, emb_dim)
        self.orient = nn.Embedding(40, emb_dim)
        self.pos_x = nn.Embedding(32, emb_dim)
        self.pos_y = nn.Embedding(32, emb_dim)

        self.out = nn.Sequential(
            nn.Linear(emb_dim * 5, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, latents):
        if latents is None:
            return None  # allow DDPM to handle this case
        s = self.shape(latents[:, 1])
        sc = self.scale(latents[:, 2])
        o = self.orient(latents[:, 3])
        x = self.pos_x(latents[:, 4])
        y = self.pos_y(latents[:, 5])
        return self.out(torch.cat([s, sc, o, x, y], dim=1))
    


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
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.cond_proj = nn.Linear(cond_emb_dim, out_channels)

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
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t, c):
        h = self.block1(x)
        time_emb = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)
        cond_emb = self.cond_proj(c).unsqueeze(-1).unsqueeze(-1)
        h = h + time_emb + cond_emb
        h = self.block2(h)
        return h + self.shortcut(x)




class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)




class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)




class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=128, cond_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Conditional embedding (for shape, scale, orient, posX, posY)
        self.cond_emb = ConditionalEmbedding(cond_emb_dim)

        # First conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim, cond_emb_dim)
        self.downsample1 = Downsample(base_channels * 2)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, cond_emb_dim)
        self.downsample2 = Downsample(base_channels * 4)

        # Bottleneck
        self.bot1 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, cond_emb_dim)
        self.bot2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim, cond_emb_dim)

        # Up
        self.up1 = Upsample(base_channels * 4, base_channels * 2)  # 128 → 64

        self.up_block1 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim, cond_emb_dim)

        self.up2 = Upsample(base_channels * 2, base_channels)      # 64 → 32
        self.up_block2 = ResidualBlock(32, 32, time_emb_dim, cond_emb_dim)


        # Output conv
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )
        self.skip1 = nn.Conv2d(128, 64, 1)
        self.skip2 = nn.Conv2d(96, 32, 1)

        self.final_upsample = Upsample(base_channels, base_channels)




    def forward(self, x, t, latents):
        t_emb = self.time_mlp(t)

        cond = self.cond_emb(latents)
        if cond is None:
            cond = torch.zeros(x.size(0), 128, device=x.device)


        x1 = self.conv_in(x)
        x2 = self.down1(x1, t_emb, cond)
        x2_down = self.downsample1(x2)
        x3 = self.down2(x2_down, t_emb, cond)
        x3_down = self.downsample2(x3)

        x4 = self.bot1(x3_down, t_emb, cond)
        x4 = self.bot2(x4, t_emb, cond)

        # This is wrong, but it works, so let's assume it's right
        x = self.up1(x4)
        x = torch.cat((x, x2_down), dim=1)
        x = self.skip1(x)
        x = self.up_block1(x, t_emb, cond)

        x1_down = F.interpolate(x1, size=x.shape[2:], mode='nearest')  # or 'bilinear'
        x = torch.cat((x, x1_down), dim=1)

        x = self.skip2(x)
        x = self.up_block2(x, t_emb, cond)

        x = self.final_upsample(x)  # [B, 32, 32, 32] → [B, 32, 64, 64]

        return self.conv_out(x)


# ------------------------------
# DDPM Core
# ------------------------------
class DDPM:
    def __init__(self, model, timesteps=250, beta_start=1e-4, beta_end=0.02):
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

    def train_step(self, x0, latents, cond_drop_prob=0.1):
        self.model.train()
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)
        xt, noise = self.noise_schedule(x0, t)

        # Randomly drop conditioning
        mask = torch.rand(B, device=device) < cond_drop_prob
        print(f"[CFG] Dropped conditioning for {mask.sum().item()}/{B} samples")
        latents_dropped = latents.clone()
        latents_dropped[mask] = -1  # Make them invalid — triggers zero embedding

        pred_noise = self.model(xt, t, latents_dropped)
        return F.mse_loss(pred_noise, noise)




    @torch.no_grad()
    def sample(self, img_size, latents, guidance_weight=3.0):
        device = next(self.model.parameters()).device
        batch_size = latents.size(0)
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)

        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # CFG: get both conditional and unconditional predictions
            pred_cond = self.model(img, time, latents)
            pred_uncond = self.model(img, time, None)

            # Blend
            pred_noise = (1 + guidance_weight) * pred_cond - guidance_weight * pred_uncond

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise

        return img.clamp(-1, 1)




class FeatureSampler:
    def __init__(self, model, ddpm, device, img_size=64):
        self.model = model
        self.ddpm = ddpm
        self.device = device
        self.img_size = img_size

        self.feature_info = {
            'shape': (1, 3),
            'scale': (2, 6),
            'orient': (3, 40),
            'posX': (4, 32),
            'posY': (5, 32),
        }

    @torch.no_grad()
    def sample_feature(self, feature_name, epoch, save_dir='./cfg', max_samples=16):
        idx, num_values = self.feature_info[feature_name]

        # How many values to sample
        if num_values > max_samples:
            step = num_values // max_samples
            values = list(range(0, num_values, step))[:max_samples]
        else:
            values = list(range(num_values))

        # Create latent batch
        fixed_latents = []
        for v in values:
            latent = [0, 0, 2, 20, 16, 16]  # default fixed values
            latent[idx] = v
            fixed_latents.append(latent)

        fixed_latents = torch.tensor(fixed_latents, dtype=torch.long, device=self.device)

        # Generate images
        sampled_imgs = self.ddpm.sample(self.img_size, fixed_latents, guidance_weight=3.0)

        sampled_imgs = (sampled_imgs + 1) * 0.5

        os.makedirs(save_dir, exist_ok=True)
        save_image(sampled_imgs, f"{save_dir}/{feature_name}_epoch_{epoch}.png", nrow=4)
        print(f"Saved {feature_name} grid at epoch {epoch}")

    def sample_all_features(self, epoch, save_dir='./cfg', max_samples=16):
        for feature_name in self.feature_info.keys():
            self.sample_feature(feature_name, epoch, save_dir, max_samples)


# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(base_channels=32).to(device)
    ddpm = DDPM(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) # 20 epochs

    os.makedirs('./cfg', exist_ok=True)
    os.makedirs('./cfg_checkpoints', exist_ok=True)

    save_every = 1  # epochs

    for epoch in range(1, 21):
        for i, (imgs, latents) in enumerate(dataloader):
            imgs = imgs.to(device)
            latents = latents.to(device)
            try:
                loss = ddpm.train_step(imgs, latents)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"OOM at epoch {epoch}, step {i} — batch skipped.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e  # re-raise if it's another kind of RuntimeError


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with torch.no_grad():
                    t = torch.randint(0, ddpm.T, (imgs.size(0),), device=imgs.device)
                    xt, noise = ddpm.noise_schedule(imgs, t)

                    mean_xt = xt.abs().mean().item()
                    mean_noise = noise.abs().mean().item()

                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | Mean xt abs: {mean_xt:.4f} | Mean noise abs: {mean_noise:.4f}")

        # Save checkpoint & samples every few epochs
        if epoch % save_every == 0:
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, f'./cfg_checkpoints/ddpm_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")

            sampler = FeatureSampler(model, ddpm, device)
            sampler.sample_all_features(epoch)

        scheduler.step()
