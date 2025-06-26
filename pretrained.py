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
from stupid_encoder import LatentClassifier, LatentEncoder


torch.manual_seed(0)
np.random.seed(0)


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
        #img = TF.resize(img, [32, 32], interpolation=TF.InterpolationMode.BILINEAR)
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
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim=None):
        super().__init__()
        self.use_z = cond_dim is not None
        if self.use_z:
            self.cond_proj = nn.Linear(cond_dim, out_channels)

        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

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

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    
    def forward(self, x, t, z=None):
        h = self.block1(x)
        h = h + self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        if self.use_z and z is not None:
            h = h + self.cond_proj(z).unsqueeze(-1).unsqueeze(-1)
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
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, cond_dim=11):
        super().__init__()
        self.cond_dim = cond_dim
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

        self.down1 = ResidualBlock(c1, c2, time_emb_dim, cond_dim)
        self.downsample1 = Downsample(c2)

        self.down2 = ResidualBlock(c2, c3, time_emb_dim, cond_dim)
        self.downsample2 = Downsample(c3)

        self.bot1 = ResidualBlock(c3, c3, time_emb_dim, cond_dim)
        self.bot2 = ResidualBlock(c3, c3, time_emb_dim, cond_dim)

        self.up1 = Upsample(c3)
        self.up_block1 = ResidualBlock(c2 + c2, c2, time_emb_dim, cond_dim)

        self.up2 = Upsample(c2)
        self.up_block2 = ResidualBlock(c1 + c1, c1, time_emb_dim, cond_dim)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, 3, padding=1)
        )

    def forward(self, x, t, z=None):
        t_emb = self.time_mlp(t)
        if z is None:
            z = torch.zeros(x.size(0), self.cond_dim, device=x.device)

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

    def train_step(self, x0, encoder=None, classifier=None, lambda_dis=1.0):
        self.model.train()
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)
        xt, noise = self.noise_schedule(x0, t)

        with torch.no_grad():
            z_dis = encoder(x0, normalize=True)

        pred_noise = self.model(xt, t, z_dis)
        loss_recon = F.mse_loss(pred_noise, noise)

        # --- Disentanglement loss ---
        loss_dis = torch.tensor(0.0, device=device)
        if classifier is not None:
            # Decode predicted z back into image
            x_recon = xt - pred_noise  # predicted x0 approximation (noisy, but works)
            z_recon = encoder(x_recon, normalize=True)
            preds = classifier(z_recon)
            with torch.no_grad():
                targets = classifier(z_dis)

            # Cross-entropy between predicted classes
            for pred, target in zip(preds, targets):
                target_class = target.argmax(dim=1)
                loss_dis += F.cross_entropy(pred, target_class)

            loss_dis = loss_dis / len(preds)

        total_loss = loss_recon + lambda_dis * loss_dis
        return total_loss


    @torch.no_grad()
    def sample(self, img_size, batch_size=256):
        device = next(self.model.parameters()).device
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)
        z = torch.zeros(batch_size, self.model.cond_dim, device=device) 
        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.model(img, time, z)
            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]
            noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise
        return img.clamp(-1, 1)
    

    @torch.no_grad()
    def sample_from_z(self, z, img_size):
        B = z.size(0)
        device = z.device
        img = torch.randn(B, 1, img_size, img_size, device=device)
        for t in reversed(range(self.T)):
            time = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = self.model(img, time, z)
            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]
            noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise
        return img.clamp(-1, 1)





@torch.no_grad()
def full_latent_traversal(ddpm, img_size=64, n_steps=7, scale=3.0, epoch=0):
    import torchvision.utils as vutils
    device = next(ddpm.model.parameters()).device

    z_dim = ddpm.model.cond_dim
    z_base = torch.zeros(1, z_dim, device=device)
    steps = torch.linspace(-scale, scale, n_steps, device=device)

    # Mapping from factor to latent dim slice
    FACTOR_SLICES = {
        'shape': (0, 1),
        'scale': (1, 3),
        'rotation': (3, 7),
        'pos_x': (7, 9),
        'pos_y': (9, 11)
    }

    all_imgs = []

    for name, (start, end) in FACTOR_SLICES.items():
        z_traversal = z_base.repeat(n_steps, 1)
        for i in range(start, end):
            z_traversal[:, i] = steps  # same steps applied to each subdim
        imgs = ddpm.sample_from_z(z_traversal, img_size)
        imgs = (imgs + 1) * 0.5  # [-1, 1] → [0, 1]
        all_imgs.append(imgs)

    # Stack rows vertically into a big grid (5 rows × n_steps)
    grid = torch.cat(all_imgs, dim=0)
    save_path = f'./pretrained_samples/latent_grid_epoch_{epoch}.png'
    vutils.save_image(grid, save_path, nrow=n_steps)




# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8)


    enc = LatentEncoder(z_dim=11).to(device)
    clf = LatentClassifier().to(device)
    state = torch.load('pretrained.pt', map_location=device)
    enc.load_state_dict(state['encoder'])
    clf.load_state_dict(state['classifier'])

    # Freeze them
    enc.eval()
    clf.eval()
    for p in enc.parameters():
        p.requires_grad = False
    for p in clf.parameters():
        p.requires_grad = False



    # Build and compile once
    model = UNet(base_channels=64).to(device)
    model = torch.compile(model)

    # Create EMA once
    ema_model = deepcopy(model)
    ema_decay = 0.995

    ddpm = DDPM(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    os.makedirs('./pretrained_checkpoints', exist_ok=True)
    os.makedirs('./pretrained_samples', exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    save_every = 1

    fixed_noise = torch.randn(16, 1, 64, 64, device=device)

    start_epoch = 1  # default starting point

    checkpoint = None
    start_epoch = 1

    checkpoint_dir = './pretrained_checkpoints'
    latest_ckpt = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if latest_ckpt:
        path = os.path.join(checkpoint_dir, latest_ckpt[-1])
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if 'encoder_state_dict' in checkpoint:
            enc.load_state_dict(checkpoint['encoder_state_dict'])
            clf.load_state_dict(checkpoint['classifier_state_dict'])
            enc.eval()
            clf.eval()
            for p in enc.parameters():
                p.requires_grad = False
            for p in clf.parameters():
                p.requires_grad = False
        else:
            print("⚠️  Warning: encoder/classifier not found in checkpoint. Falling back to pretrained.pt")
            pretrained = torch.load('pretrained.pt', map_location=device)
            enc.load_state_dict(pretrained['encoder'])
            clf.load_state_dict(pretrained['classifier'])




    for epoch in range(start_epoch, 100):
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss = ddpm.train_step(batch, encoder=enc, classifier=clf, lambda_dis=1.0)

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
                'encoder_state_dict': enc.state_dict(),  
                'classifier_state_dict': clf.state_dict(), 
                'epoch': epoch,
            }, f'./pretrained_checkpoints/ddpm_epoch_{epoch}.pt')


            print(f"Saved checkpoint at epoch {epoch}")

            # Use EMA model temporarily for sampling
            ddpm.model = ema_model
            with torch.no_grad():
                z = torch.zeros(16, ddpm.model.cond_dim, device=device)
                sampled_imgs = ddpm.sample_from_z(z, img_size=64)
                # === Latent prediction check ===
                with torch.no_grad():
                    # Get predicted latents from generated images
                    z_pred = enc(sampled_imgs, normalize=True)
                    preds = clf(z_pred)

                    targets = clf(z)  # should decode same as z
                    correct = []

                    for pred, target in zip(preds, targets):
                        pred_class = pred.argmax(dim=1)
                        target_class = target.argmax(dim=1)
                        correct.append((pred_class == target_class).float().mean().item())

                    acc_names = ['shape', 'scale', 'rotation', 'pos_x', 'pos_y']
                    for name, acc in zip(acc_names, correct):
                        print(f"DDPM sample latent {name} accuracy: {acc:.3f}")


            full_latent_traversal(ddpm, img_size=64, epoch=epoch)
            print(f"Saved samples at epoch {epoch}")


            # === Sample Quality Metrics ===
            with torch.no_grad():
                # Use a small fixed batch of real samples
                real_batch = next(iter(dataloader))[:16].to(device)
                gen_batch = sampled_imgs[:16].to(device)

                # Resize real batch to match generated image size (64x64)
                real_batch = F.interpolate(real_batch, size=(64, 64), mode='bilinear', align_corners=False)

                # L1 Distance (in [-1, 1] space)
                l1 = F.l1_loss(gen_batch, real_batch).item()


                # Binariness score
                binariness = ((gen_batch < 0.05) | (gen_batch > 0.95)).float().mean().item()

                print(f"Sample Quality @ Epoch {epoch} | L1: {l1:.4f} | Binariness: {binariness:.4f}")

                # Save to text file
                with open("./pretrained_samples/sample_quality.txt", "a") as f:
                    f.write(f"{epoch},{l1:.4f},{binariness:.4f}\n")

                pure_black = (sampled_imgs.view(sampled_imgs.size(0), -1).max(dim=1).values < 0.05).float().mean().item()
                print(f"Pure black ratio: {pure_black:.4f}")


            # Switch back to main model
            ddpm.model = model



        scheduler.step()                                            
