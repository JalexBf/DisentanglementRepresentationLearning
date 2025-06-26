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
import torchvision.transforms.functional as TF
import csv
from pathlib import Path

torch.manual_seed(0)
np.random.seed(0)

from contextlib import contextmanager

@contextmanager
def use_ema(ddpm, ema_model):
    original_model = ddpm.model
    ddpm.model = ema_model
    try:
        yield
    finally:
        ddpm.model = original_model

# ------------------------------
# Dataset Loader
# ------------------------------
class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)
        img = TF.resize(img, [32, 32], interpolation=TF.InterpolationMode.BILINEAR)
        img = (img > 0.5).float()  # re-binarize just in case
        img = img * 2.0 - 1.0  # scale to [-1, 1]
        return img
    


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
    def __init__(self, in_channels, out_channels, time_emb_dim, P_i=None):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.P_i = P_i
        self.cond_proj = nn.Linear(P_i.size(0), out_channels) if P_i is not None else None

    def forward(self, x, t, z):
        h = self.block1(x)
        h = h + self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        if self.P_i is not None and z is not None:
            zi = F.linear(z, self.P_i.to(z.device))

            h = h + self.cond_proj(zi).unsqueeze(-1).unsqueeze(-1)
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
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, P_list=None):
        super().__init__()
        self.dim_per_factor = [4, 4, 8, 6, 6]
        self.total_cond_dim = sum(self.dim_per_factor)
        assert P_list is not None, "You must pass P_list explicitly."
        self.P_list = P_list

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.conv_in = nn.Conv2d(in_channels, c1, 3, padding=1)

        self.down1 = ResidualBlock(c1, c2, time_emb_dim, self.P_list[0])  # shape
        self.downsample1 = Downsample(c2)

        self.down2 = ResidualBlock(c2, c3, time_emb_dim, self.P_list[1])  # scale
        self.downsample2 = Downsample(c3)

        self.bot1 = ResidualBlock(c3, c3, time_emb_dim, self.P_list[2])   # orientation
        self.bot2 = ResidualBlock(c3, c3, time_emb_dim, self.P_list[2])   # orientation again

        self.up1 = Upsample(c3)
        self.up_block1 = ResidualBlock(c2 + c2, c2, time_emb_dim, self.P_list[3])  # posX

        self.up2 = Upsample(c2)
        self.up_block2 = ResidualBlock(c1 + c1, c1, time_emb_dim, self.P_list[4])  # posY

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, 3, padding=1)
        )

    def forward(self, x, t, z=None):
        t_emb = self.time_mlp(t)
        if z is None:
            z = torch.zeros(x.size(0), self.total_cond_dim, device=x.device)

        x1 = self.conv_in(x)
        x2 = self.down1(x1, t_emb, z)
        x2_down = self.downsample1(x2)
        x3 = self.down2(x2_down, t_emb, z)
        x3_down = self.downsample2(x3)
        x4 = self.bot1(x3_down, t_emb, z)
        x4 = self.bot2(x4, t_emb, z)
        x = self.up1(x4)
        x = torch.cat((x, x2_down), dim=1)
        x = self.up_block1(x, t_emb, z)
        x = self.up2(x)
        x = torch.cat((x, x1), dim=1)
        x = self.up_block2(x, t_emb, z)
        return self.conv_out(x)





# ------------------------------
# DDPM Core
# ------------------------------
class DDPM:
    def __init__(self, model, timesteps=250, beta_start=1e-4, beta_end=0.01):
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
    
    @torch.no_grad()
    def p_sample(self, x, t, z):
        pred_noise = self.model(x, t, z)

        beta = self.beta[t].reshape(-1, 1, 1, 1)
        alpha = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_hat = self.alpha_hat[t].reshape(-1, 1, 1, 1)

        sqrt_alpha = alpha.sqrt()
        sqrt_alpha_hat = alpha_hat.sqrt()
        sqrt_one_minus_alpha_hat = (1 - alpha_hat).sqrt()

        coef1 = 1 / sqrt_alpha
        coef2 = (1 - alpha) / sqrt_one_minus_alpha_hat

        noise = torch.randn_like(x)
        noise[t == 0] = 0  # no noise at final step

        x_prev = coef1 * (x - coef2 * pred_noise) + beta.sqrt() * noise
        return x_prev



    def train_step(self, x0, epoch):
        self.model.train()
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)
        xt, noise = self.noise_schedule(x0, t)

        dim_per_factor = [4, 4, 8, 6, 6]
        total_dim = sum(dim_per_factor)
        if epoch < 5:
            z = torch.zeros(B, total_dim, device=device)
        else:
            z = torch.randn(B, total_dim, device=device)


        def get_factor_slices():
            dims = [4, 4, 8, 6, 6]
            slices = []
            start = 0
            for d in dims:
                end = start + d
                slices.append((start, end))
                start = end
            return slices

        with autocast():
            pred_noise = self.model(xt, t, z)
            mse_loss = F.mse_loss(pred_noise, noise)

            # --- LEAKAGE PENALTY ---
            leakage_loss = 0.0
            with torch.no_grad():
                slices = get_factor_slices()
                for i, (start, end) in enumerate(slices):
                    z_varied = z.clone()
                    z_varied[:, start:end] += torch.randn(B, end - start, device=z.device)
                    pred_var = self.model(xt, t, z_varied)
                    delta = (pred_var - pred_noise).abs()

                    for j, (s2, e2) in enumerate(slices):
                        if i != j:
                            leakage_loss += delta.mean()

                leakage_loss /= (len(slices) * (len(slices) - 1))  # Normalised


            if epoch < 5:
                loss = mse_loss
                leakage_loss = torch.tensor(0.0, device=x0.device)  # ensure it's logged
            else:
                warmup_epoch = epoch - 5
                weight = 0.1 * (1 - math.cos(math.pi * min(warmup_epoch, 10) / 10)) / 2
                loss = mse_loss + weight * leakage_loss



        return loss, mse_loss.item(), leakage_loss.item()



    @torch.no_grad()
    def sample(self, img_size, batch_size=256):
        device = next(self.model.parameters()).device
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)
        z = torch.zeros(batch_size, 28, device=device) 
        with autocast(): 
            for t in reversed(range(self.T)):
                time = torch.full((batch_size,), t, device=device, dtype=torch.long)
                img = self.p_sample(img, time, z)

        return img.clamp(-1, 1)




@torch.no_grad()
def debug_denoising(ddpm, dataloader):
    batch = next(iter(dataloader))[:8].to(next(ddpm.model.parameters()).device)
    t = torch.randint(200, 250, (batch.size(0),), device=batch.device)
    xt, noise = ddpm.noise_schedule(batch, t)
    z = torch.zeros(batch.size(0), 28, device=batch.device)
    x_denoised = ddpm.p_sample(xt, t, z)

    save_image((batch + 1) * 0.5, "debug_gt.png", nrow=4)
    save_image((xt + 1) * 0.5, "debug_noisy.png", nrow=4)
    save_image((x_denoised + 1) * 0.5, "debug_denoised.png", nrow=4)
    print("Saved debug_gt.png / debug_noisy.png / debug_denoised.png")




def create_orthogonal_subspaces(dim_per_factor=[4, 4, 8, 6, 6]):
    total_dim = sum(dim_per_factor)
    # Generate a random orthogonal matrix
    Q, _ = torch.linalg.qr(torch.randn(total_dim, total_dim))
    subspaces = []
    start = 0
    for d in dim_per_factor:
        end = start + d
        P = Q[start:end, :]  # d x total_dim
        subspaces.append(P)
        start = end
    return subspaces






def append_sensitivity_log(epoch, sensitivities, path="./orthogonal_samples/orthogonal_leakage.csv"):
        file_exists = os.path.isfile(path)
        with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["epoch", "shape", "scale", "orientation", "posX", "posY"])
            writer.writerow([epoch] + [f"{val:.5f}" for val in sensitivities])


@torch.no_grad()
def test_orthogonal_generation(ddpm, steps_per_factor=8, save_path='orthogonal_test.png', epoch=0):
    device = next(ddpm.model.parameters()).device
    z_dim = 28
    z = torch.zeros(steps_per_factor * 5, z_dim, device=device)

    # Vary each factor one at a time
    factor_slices = {
        'shape':       slice(0, 4),
        'scale':       slice(4, 8),
        'orientation': slice(8, 16),
        'posX':        slice(16, 22),
        'posY':        slice(22, 28),
    }

    for i, (factor, sl) in enumerate(factor_slices.items()):
        steps = torch.linspace(-3, 3, steps_per_factor).unsqueeze(1).to(device)
        z[i * steps_per_factor:(i + 1) * steps_per_factor, sl] = steps.repeat(1, sl.stop - sl.start)

    # Generate images
    steps = steps_per_factor
    fixed_noise = torch.randn(1, 1, 32, 32, device=device).repeat(steps * 5, 1, 1, 1)
    img = fixed_noise.clone()

    for t in reversed(range(ddpm.T)):
        time = torch.full((img.size(0),), t, device=device, dtype=torch.long)
        img = ddpm.p_sample(img, time, z)

    samples = (img.clamp(-1, 1) + 1) * 0.5
    save_image(samples, save_path, nrow=steps_per_factor)
    print(f"Saved orthogonality test image: {save_path}")

    samples_reshaped = samples.view(5, steps_per_factor, 1, 32, 32)
    diffs = (samples_reshaped[:, 1:] - samples_reshaped[:, :-1]).abs().mean(dim=(2, 3, 4))
    for i, d in enumerate(diffs.mean(dim=1).tolist()):
        print(f"[Factor {i}] Mean pixel sensitivity across steps: {d:.4f}")
    sensitivity_vals = diffs.mean(dim=1).tolist()
    append_sensitivity_log(epoch, sensitivity_vals)


    with torch.no_grad():
        cross_L1 = torch.zeros(5, 5, device=device)
        rows = samples.view(5, steps_per_factor, 1, 32, 32)
        for i in range(5):
            for j in range(5):
                if i != j:
                    cross_L1[i, j] = F.l1_loss(rows[i], rows[j])
        print("[Cross-factor L1 leakage matrix]")
        print(cross_L1.cpu().numpy().round(4))




@torch.no_grad()
def measure_factor_effect(ddpm, P_list, n_samples=8, save_prefix="factor_effect"):
    device = next(ddpm.model.parameters()).device
    total_dim = sum(p.size(0) for p in P_list)
    base_noise = torch.randn(n_samples, 1, 32, 32, device=device)
    z = torch.zeros(n_samples, total_dim, device=device)

    for i, P in enumerate(P_list):
        sub_dim = P.size(0)
        steps = torch.linspace(-3, 3, n_samples).unsqueeze(1).to(device)
        varied = steps.repeat(1, sub_dim)
        full_z = torch.zeros(n_samples, total_dim, device=device)
        full_z[:, :] = 0
        start = sum(p.size(0) for p in P_list[:i])
        end = start + P.size(0)
        full_z[:, start:end] = varied


                    
        img = base_noise.clone()
        for t in reversed(range(ddpm.T)):
            time = torch.full((n_samples,), t, device=device, dtype=torch.long)
            img = ddpm.p_sample(img, time, full_z)


        img = (img.clamp(-1, 1) + 1) * 0.5
        save_image(img, f"{save_prefix}_factor_{i}.png", nrow=n_samples)
        print(f"Saved diagnostic: {save_prefix}_factor_{i}.png")




# ------------------------------
# Training Loop
# ------------------------------
# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)

    dim_per_factor = [4, 4, 8, 6, 6]
    P_path = Path('./orthogonal_checkpoints/P_list.pt')
    checkpoint_dir = './orthogonal_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('./orthogonal_samples', exist_ok=True)

    checkpoint = None
    start_epoch = 1
    latest_ckpt = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.startswith('ddpm_epoch_') and f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if latest_ckpt:
        path = os.path.join(checkpoint_dir, latest_ckpt[-1])
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1

    # === Handle P_list ===
    if checkpoint is not None and 'P_list' in checkpoint:
        P_list = checkpoint['P_list']
        print("Loaded P_list from checkpoint.")
    elif P_path.exists():
        P_list = torch.load(P_path)
        print("Loaded P_list from file.")
    else:
        P_list = create_orthogonal_subspaces(dim_per_factor)
        torch.save(P_list, P_path)
        print("Saved new P_list.")

 
    model = UNet(base_channels=64, P_list=P_list).to(device)
    model = torch.compile(model)

    ema_model = deepcopy(model)
    ema_decay = 0.995

    if checkpoint is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
    else:
        print("No EMA state found in checkpoint. Reinitializing EMA.")

    ddpm = DDPM(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    scaler = GradScaler()
    save_every = 1

    for epoch in range(start_epoch, 200):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, mse_val, leak_val = ddpm.train_step(batch, epoch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            if i % 100 == 0:
                with torch.no_grad():
                    t = torch.randint(0, ddpm.T, (batch.size(0),), device=device)
                    xt, noise = ddpm.noise_schedule(batch, t)
                    print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | MSE: {mse_val:.4f} | Leakage: {leak_val:.4f}")

        test_orthogonal_generation(ddpm, steps_per_factor=8, save_path=f'./orthogonal_samples/orthogonal_epoch_{epoch}.png', epoch=epoch)
        debug_denoising(ddpm, dataloader)


        if epoch % save_every == 0:
            measure_factor_effect(ddpm, P_list, n_samples=8, save_prefix=f"./orthogonal_samples/factor_effect_epoch_{epoch}")

            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'P_list': P_list,
            }, f'./orthogonal_checkpoints/ddpm_epoch_{epoch}.pt')

            print(f"Saved checkpoint at epoch {epoch}")

            # Use EMA model to sample
            with use_ema(ddpm, ema_model):
                with torch.no_grad():
                    real_imgs = next(iter(dataloader))[:16].to(device)
                    z = torch.randn(real_imgs.size(0), 28, device=device)
                    img = torch.randn_like(real_imgs)

                    for t in reversed(range(ddpm.T)):
                        time = torch.full((img.size(0),), t, device=device, dtype=torch.long)
                        img = ddpm.p_sample(img, time, z)


                    sampled_imgs = (img.clamp(-1, 1) + 1) * 0.5
                    save_image(sampled_imgs, f'./orthogonal_samples/sample_epoch_{epoch}.png', nrow=4)
                    print(f"Saved samples at epoch {epoch}")

                # Sample quality metrics
                real_batch = next(iter(dataloader))[:16].to(device)
                gen_batch = sampled_imgs[:16].to(device)
                l1 = F.l1_loss(gen_batch * 2 - 1, real_batch).item()
                binariness = ((gen_batch < 0.05) | (gen_batch > 0.95)).float().mean().item()
                pure_black = (sampled_imgs.view(sampled_imgs.size(0), -1).max(dim=1).values < 0.05).float().mean().item()
                print(f"Sample Quality @ Epoch {epoch} | L1: {l1:.4f} | Binariness: {binariness:.4f} | Pure black: {pure_black:.4f}")

                with open("./orthogonal_samples/sample_quality.txt", "a") as f:
                    f.write(f"{epoch},{l1:.4f},{binariness:.4f},{pure_black:.4f}\n")

        scheduler.step()
