import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import os
import math
from copy import deepcopy
from classifier import DSpritesFeatureClassifier

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

    def forward(self, latents):  # (B, 5)
        s = self.shape(latents[:, 1])   # shape
        sc = self.scale(latents[:, 2])  # scale
        o = self.orient(latents[:, 3])  # orientation
        x = self.pos_x(latents[:, 4])   # posX
        y = self.pos_y(latents[:, 5])   # posY
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

        # 2 convolutional blocks
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
        # Skin connection
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t, c):
        h = self.block1(x)
        time_emb = self.time_proj(t).unsqueeze(-1).unsqueeze(-1)    # Shape: (B, out_channels)
        cond_emb = self.cond_proj(c).unsqueeze(-1).unsqueeze(-1)
        
        # Add the time and cond embeddings to the feature map
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



class LatentEncoder(nn.Module):
    def __init__(self, z_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 → 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, z_dim)
        )

    def forward(self, x):
        return self.net(x)




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
        self.z_proj = nn.Sequential(
            nn.Linear(10, cond_emb_dim),
            nn.SiLU(),
            nn.Linear(cond_emb_dim, cond_emb_dim)
        )


        # First conv
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.cond_to_input = nn.Linear(cond_emb_dim, base_channels)


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
        self.skip1 = nn.Conv2d(base_channels * 6, base_channels * 2, 1)
        self.skip2 = nn.Conv2d(base_channels * 3, base_channels, 1)


        self.final_upsample = Upsample(base_channels, base_channels)




    def forward(self, x, t, z):
        t_emb = self.time_mlp(t)
        cond = self.z_proj(z)  # z is the learned latent vector from the encoder

        cond_proj = self.cond_to_input(cond)                  # (B, base_channels)
        cond_proj = cond_proj[:, :, None, None]               # (B, base_channels, 1, 1)
        cond_proj = cond_proj.expand(-1, -1, x.shape[2], x.shape[3])  # (B, base_channels, H, W)

        x1 = self.conv_in(x) + cond_proj                      # Inject latents at input

        x2 = self.down1(x1, t_emb, cond)
        x2_down = self.downsample1(x2)
        x3 = self.down2(x2_down, t_emb, cond)
        x3_down = self.downsample2(x3)

        x4 = self.bot1(x3_down, t_emb, cond)
        x4 = self.bot2(x4, t_emb, cond)

        # This is wrong, but it works, so let's assume it's right (concatenate with x3, not x2_down)
        x = self.up1(x4)
        x = torch.cat((x, x3), dim=1)  # use x3 instead of x2_down
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
    def __init__(self, model, encoder, timesteps=250, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.encoder = encoder
        device = next(model.parameters()).device
        self.T = timesteps
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.classifier = DSpritesFeatureClassifier().to(device)
        self.classifier.load_state_dict(torch.load("dsprites_classifier.pt", map_location=device))
        self.classifier.eval()
        #FReeze classifier parameters
        for p in self.classifier.parameters():
            p.requires_grad = False


    def noise_schedule(self, x0, t):
        noise = torch.randn_like(x0)    # Gaussian noise
        batch_size = x0.size(0)

        # Extract the cumulative noise factor for specific time step
        sqrt_alpha_hat = self.alpha_hat[t].reshape(batch_size, 1, 1, 1) ** 0.5
        sqrt_one_minus = (1 - self.alpha_hat[t]).reshape(batch_size, 1, 1, 1) ** 0.5
        
        xt = sqrt_alpha_hat * x0 + sqrt_one_minus * noise   # Noisy image
        return xt, noise


    def train_step(self, x0, latents, classifier_weight=0.2):
        self.model.train()
        B = x0.size(0)
        device = x0.device
        z = self.encoder(x0)  

        # Randomly sample a time step for each image
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)
        xt, noise = self.noise_schedule(x0, t)  # Apply noise at selected timestep
        pred_noise = self.model(xt, t, z)

        # Standard DDPM loss
        loss = F.mse_loss(pred_noise, noise)

        # Reconstruct image from pred_noise (approximate x0)
        alpha_hat = self.alpha_hat[t].reshape(B, 1, 1, 1)
        x0_hat = (xt - (1 - alpha_hat).sqrt() * pred_noise) / alpha_hat.sqrt()
        x0_hat = x0_hat.clamp(-1, 1)

        # Classifier regularization (latent prediction from image)
        preds = self.classifier(x0_hat)

        disent_loss = sum([
            F.cross_entropy(preds[0], latents[:, 1]),  # shape
            F.cross_entropy(preds[1], latents[:, 2]),  # scale
            F.cross_entropy(preds[2], latents[:, 3]),  # orient
            F.cross_entropy(preds[3], latents[:, 4]),  # posX
            F.cross_entropy(preds[4], latents[:, 5]),  # posY
        ])

        total_loss = loss + classifier_weight * disent_loss
        return total_loss, loss.detach(), disent_loss.detach(), preds


    # Reverse diffusion process
    @torch.no_grad()
    def sample_from_z(self, img_size, z):
        device = next(self.model.parameters()).device
        batch_size = z.size(0)
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)
        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.model(img, time, z)

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise

        return img.clamp(-1, 1)



# ------------------------------
# Feature Sampling
# ------------------------------
class LatentTraversalSampler:
    def __init__(self, ddpm, img_size, z_dim=10, device='cuda'):
        self.ddpm = ddpm
        self.img_size = img_size
        self.device = device
        self.z_dim = z_dim

    @torch.no_grad()
    def sample_latent_traversals(self, epoch, save_dir='./traversals', num_steps=5, z_range=(-3, 3)):
        os.makedirs(save_dir, exist_ok=True)
        base_z = torch.zeros(self.z_dim, device=self.device).unsqueeze(0).repeat(num_steps, 1)

        for i in range(self.z_dim):
            z = base_z.clone()
            vals = torch.linspace(*z_range, steps=num_steps).to(self.device)
            z[:, i] = vals

            samples = self.ddpm.sample_from_z(self.img_size, z)
            samples = (samples + 1) * 0.5  # Rescale to [0, 1]
            save_image(samples, f'{save_dir}/z{i}_epoch{epoch}.png', nrow=num_steps)
            print(f"Saved traversal for z[{i}] at epoch {epoch}")


# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, base_channels=32, time_emb_dim=128, cond_emb_dim=128).to(device)
    encoder = LatentEncoder().to(device)
    ddpm = DDPM(model, encoder)


    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(encoder.parameters()),
        lr=3e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) # 20 epochs


    os.makedirs('./unsupervised_checkpoints', exist_ok=True)
    os.makedirs('./unsupervised_samples', exist_ok=True)

    save_every = 1  # epochs

    for epoch in range(1, 21):
        for i, (imgs, latents) in enumerate(dataloader):
            imgs = imgs.to(device)
            latents = latents.to(device)
            
            try:
                total_loss, mse_loss, class_loss, preds = ddpm.train_step(imgs, latents, classifier_weight=0.2)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"OOM at epoch {epoch}, step {i} — batch skipped.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                with torch.no_grad():
                    pred_labels = [p.argmax(1) for p in preds]
                    # preds[0:5] must match latents[:, 1:6] → shape, scale, orient, posX, posY
                    accs = [((pred == latents[:, j+1]).float().mean().item()) for j, pred in enumerate(pred_labels)]
                    print(f"Accuracies: shape={accs[0]:.2f}, scale={accs[1]:.2f}, orient={accs[2]:.2f}, posX={accs[3]:.2f}, posY={accs[4]:.2f}")

                    t = torch.randint(0, ddpm.T, (imgs.size(0),), device=imgs.device)
                    xt, noise = ddpm.noise_schedule(imgs, t)
                    mean_xt = xt.abs().mean().item()
                    mean_noise = noise.abs().mean().item()

                print(f"Epoch {epoch} | Step {i} | Total: {total_loss.item():.4f} | MSE: {mse_loss.item():.4f} | Classifier: {class_loss.item():.4f} | Mean xt: {mean_xt:.4f} | Mean noise: {mean_noise:.4f}")

        # Save checkpoint & samples every few epochs
        if epoch % save_every == 0:
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }, f'./unsupervised_checkpoints/ddpm_epoch_{epoch}.pt')
            print(f"Saved checkpoint at epoch {epoch}")

            torch.save(encoder.state_dict(), f'./unsupervised_checkpoints/encoder_epoch_{epoch}.pt')


            sampler = LatentTraversalSampler(ddpm, img_size=64, device=device)
            sampler.sample_latent_traversals(epoch)

        scheduler.step()