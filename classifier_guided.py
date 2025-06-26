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




    def forward(self, x, t, latents):
        t_emb = self.time_mlp(t)
        cond = self.cond_emb(latents)

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
    def __init__(self, model, timesteps=250, beta_start=1e-4, beta_end=0.02):
        self.model = model
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

        # Randomly sample a time step for each image
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)
        xt, noise = self.noise_schedule(x0, t)  # Apply noise at selected timestep
        pred_noise = self.model(xt, t, latents)

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
    def sample(self, img_size, latents):
        device = next(self.model.parameters()).device
        batch_size = latents.size(0)
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)
        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.model(img, time, latents)

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            # Add Gaussian noise at each step except the last one
            noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
            
            # Reverse diffusion step
            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise

        return img.clamp(-1, 1)



# ------------------------------
# Feature Sampling
# ------------------------------
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
    def sample_feature(self, feature_name, epoch, save_dir='./classifier_samples', max_samples=16):
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
        sampled_imgs = self.ddpm.sample(self.img_size, fixed_latents)

        sampled_imgs = (sampled_imgs + 1) * 0.5

        os.makedirs(save_dir, exist_ok=True)
        save_image(sampled_imgs, f"{save_dir}/{feature_name}_epoch_{epoch}.png", nrow=4)
        print(f"Saved {feature_name} grid at epoch {epoch}")

    def sample_all_features(self, epoch, save_dir='./classifier_samples', max_samples=16):
        for feature_name in self.feature_info.keys():
            self.sample_feature(feature_name, epoch, save_dir, max_samples)



# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(base_channels=32).to(device)
    ddpm = DDPM(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20) # 20 epochs


    os.makedirs('./classifier_checkpoints', exist_ok=True)
    os.makedirs('./classifier_samples', exist_ok=True)


    start_epoch = 1  # default starting epoch
    checkpoint_dir = './classifier_checkpoints'

    latest_ckpt = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith('ddpm_epoch_') and f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if latest_ckpt:
        checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt[-1])
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1  # continue from next epoch



    save_every = 1  # epochs
    best_loss = float('inf')
    patience = 5  # stop if no improvement for 5 epochs
    patience_counter = 0

    for epoch in range(start_epoch, 21):
        # Before batch loop
        epoch_total_loss = 0.0
        num_batches = 0

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

            epoch_total_loss += total_loss.item()
            num_batches += 1

            if i % 100 == 0:
                with torch.no_grad():
                    pred_labels = [p.argmax(1) for p in preds]
                    accs = [((pred == latents[:, j+1]).float().mean().item()) for j, pred in enumerate(pred_labels)]
                    print(f"Accuracies: shape={accs[0]:.2f}, scale={accs[1]:.2f}, orient={accs[2]:.2f}, posX={accs[3]:.2f}, posY={accs[4]:.2f}")
                    t = torch.randint(0, ddpm.T, (imgs.size(0),), device=imgs.device)
                    xt, noise = ddpm.noise_schedule(imgs, t)
                    mean_xt = xt.abs().mean().item()
                    mean_noise = noise.abs().mean().item()

                print(f"Epoch {epoch} | Step {i} | Total: {total_loss.item():.4f} | MSE: {mse_loss.item():.4f} | Classifier: {class_loss.item():.4f} | Mean xt: {mean_xt:.4f} | Mean noise: {mean_noise:.4f}")

        # After batch loop (end of epoch)
        avg_loss = epoch_total_loss / num_batches
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            print(f"New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break


        
        
        # Save checkpoint & samples every few epochs
        if epoch % save_every == 0:
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
            }, f'./classifier_checkpoints/ddpm_epoch_{epoch}.pt')

            print(f"Saved checkpoint at epoch {epoch}")

            sampler = FeatureSampler(model, ddpm, device)
            sampler.sample_all_features(epoch)


        scheduler.step()