import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import os
import math
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from PIL import ImageDraw, ImageFont
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

torch.manual_seed(0)
np.random.seed(0)


# ------------------------------
# Dataset Loader
# ------------------------------
class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']
        self.latents_classes = self.data['latents_classes']

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)
        img = TF.resize(img, [32, 32], interpolation=TF.InterpolationMode.BILINEAR)
        img = (img > 0.5).float()  # re-binarize just in case
        img = img * 2.0 - 1.0  # scale to [-1, 1]
        labels = torch.from_numpy(self.latents_classes[idx]).long()
        return img, labels



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

    def forward(self, x, t, z=None):
        h = self.block1(x)
        t_emb = self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        if self.use_z and z is not None:
            z_emb = self.cond_proj(z).unsqueeze(-1).unsqueeze(-1)
            h = h + z_emb
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
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, cond_dim=None):
        super().__init__()
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
    def __init__(self, model, encoder=None, classifier=None, timesteps=250, beta_start=1e-4, beta_end=0.01):
        self.model = model
        self.encoder = encoder
        self.classifier = classifier
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

    def train_step(self, x0, gt):
        self.model.train()
        B = x0.size(0)
        device = x0.device

        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)

        with torch.cuda.amp.autocast():
            z = torch.zeros(x0.size(0), 5, device=x0.device)


            #print("z std:", z.std(dim=0))
            if torch.isnan(z).any():
                print("NaNs in z")
            if torch.isinf(z).any():
                print("Infs in z")

            if torch.isnan(z).any():
                print("NaNs in z")
            if torch.isinf(z).any():
                print("Infs in z")


            xt, noise = self.noise_schedule(x0, t)
            pred_noise = self.model(xt, t, z)
            loss_recon = F.mse_loss(pred_noise, noise)

            # If encoder is frozen, skip loss_cls and loss_tc
            if self.encoder.parameters().__next__().requires_grad:
                if self.classifier is not None:
                    pred_factors = self.classifier(z)
                    loss_cls = sum(F.cross_entropy(pred_factors[i], gt[:, i]) for i in range(len(pred_factors))) / 5
                else:
                    loss_cls = torch.tensor(0.0, device=device)

                def total_correlation(z):
                    if torch.isnan(z).any() or torch.isinf(z).any():
                        print("[TC] z has NaNs or Infs before computing covariance")
                        return torch.tensor(float('nan'), device=z.device)

                    z_centered = z - z.mean(dim=0)
                    cov = z_centered.T @ z_centered / (z.size(0) - 1)

                    if torch.isnan(cov).any() or torch.isinf(cov).any():
                        print("[TC] covariance matrix has NaNs or Infs")
                        return torch.tensor(float('nan'), device=z.device)

                    off_diag = cov - torch.diag(torch.diag(cov))
                    return off_diag.abs().mean()

                loss_tc = total_correlation(z)

            else:
                loss_cls = torch.tensor(0.0, device=device)
                loss_tc = torch.tensor(0.0, device=device)

        return loss_recon, loss_cls, loss_tc


                


    @torch.no_grad()
    def sample(self, img_size, batch_size=512, z=None):
        device = next(self.model.parameters()).device
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)

        if z is None:
            z = torch.randn(batch_size, 5, device=device)

        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = self.model(img, time, z)

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            if t > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise

        return img.clamp(-1, 1)





class LatentEncoder(nn.Module):
    def __init__(self, z_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # 32 → 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 16 → 8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 8 → 4
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), # 4 → 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z / z.std(dim=0, keepdim=True)


#=============================================================



# latent.py

class LatentClassifier(nn.Module):
    def __init__(self, z_dim=5):
        super().__init__()
        self.shape_head = nn.Linear(z_dim, 3)
        self.scale_head = nn.Linear(z_dim, 6)
        self.orient_head = nn.Linear(z_dim, 40)  
        self.pos_x_head = nn.Linear(z_dim, 32)
        self.pos_y_head = nn.Linear(z_dim, 32)

    def forward(self, z):
        return [
            self.shape_head(z),
            self.scale_head(z),
            self.orient_head(z),  
            self.pos_x_head(z),
            self.pos_y_head(z),
        ]






@torch.no_grad()
def latent_traversal(ddpm, z_dim=5, steps=10, img_size=64, fixed_z=None):
    device = next(ddpm.model.parameters()).device
    if fixed_z is None:
        fixed_z = torch.randn(1, z_dim, device=device)
    else:
        fixed_z = fixed_z.detach()


    all_imgs = []

    for i in range(z_dim):
        z_batch = fixed_z.repeat(steps, 1)
        linspace = torch.linspace(-3, 3, steps).to(device)
        z_batch[:, i] = linspace  # vary only dim i

        imgs = ddpm.sample(img_size=img_size, batch_size=steps, z=z_batch)
        print("z_batch stats:", z_batch.mean().item(), z_batch.std().item())
        imgs = (imgs + 1) * 0.5  # scale to [0, 1]
        
        annotated = []
        for j in range(steps):
            img = imgs[j].detach()  # ← detach here
            text = f"z{i}: {z_batch[j, i].item():.2f}"
            img_rgb = TF.to_pil_image(img.expand(3, -1, -1))

            draw = ImageDraw.Draw(img_rgb)
            draw.text((3, 3), text, fill=(255, 255, 255))  # white text
            annotated.append(TF.to_tensor(img_rgb))

        all_imgs.append(torch.stack(annotated))  


    return all_imgs


# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)

    model = UNet(base_channels=64, cond_dim=5).to(device)
    encoder = LatentEncoder(z_dim=5).to(device)
    #encoder = torch.compile(encoder)
    classifier = LatentClassifier(z_dim=5).to(device)
    #classifier = torch.compile(classifier)

    ema_model = deepcopy(model)
    ema_encoder = deepcopy(encoder) 

    ema_decay = 0.995

    ddpm = DDPM(model, encoder=encoder, classifier=classifier)


    optimizer = torch.optim.Adam(
        list(model.parameters()) +
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=1e-5
    )


    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    os.makedirs('./nopretraining_checkpoints', exist_ok=True)
    os.makedirs('./nopretraining_samples', exist_ok=True)



    start_epoch = 1  # default

    # Check if we should resume
    checkpoint_dir = './nopretraining_checkpoints'
    latest_ckpt = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    if latest_ckpt:
        path = os.path.join(checkpoint_dir, latest_ckpt[-1])
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if 'ema_state_dict' in checkpoint:
            ema_model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in checkpoint['ema_state_dict'].items()})
            ema_encoder.load_state_dict(checkpoint['ema_encoder_state_dict'])
        else:
            print("No EMA state found in checkpoint. Reinitializing EMA.")

    else:
        # If no checkpoint to resume, load pretrained
        pretrained_path = './simple_checkpoints/ddpm_epoch_20.pt'
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained DDPM from {pretrained_path}")
            pretrained_ckpt = torch.load(pretrained_path, map_location=device)

            # Remove _orig_mod. if model was compiled with torch.compile()
            pretrained_dict = {
                k.replace('_orig_mod.', ''): v
                for k, v in pretrained_ckpt['model_state_dict'].items()
            }

            model_dict = model.state_dict()
            compatible_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }

            model.load_state_dict(compatible_dict, strict=False)
            ema_model.load_state_dict(model.state_dict())  # Optional: sync EMA to initial model
            print(f"Loaded {len(compatible_dict)} layers from pretrained DDPM.")
        else:
            print(f"[WARN] Pretrained checkpoint not found at {pretrained_path}. Skipping.")





    scaler = torch.amp.GradScaler()
    save_every = 1

    fixed_noise = torch.randn(16, 1, 64, 64, device=device)


    checkpoint = None
    for epoch in range(start_epoch, 200):
        for i, (batch, labels) in enumerate(dataloader):
            batch = batch.to(device)
            labels = labels[:, 1:6].to(device)  # Skip latent dim 0 (color — fixed in dSprites)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                t = torch.randint(0, ddpm.T, (batch.size(0),), device=batch.device)
                xt, noise = ddpm.noise_schedule(batch, t)

                # ===== Set requires_grad properly outside the loss logic =====
                if 5 <= epoch < 10:
                    for p in model.parameters():
                        p.requires_grad = False  # freeze DDPM
                    for p in encoder.parameters():
                        p.requires_grad = True
                    for p in classifier.parameters():
                        p.requires_grad = True
                else:
                    for p in model.parameters():
                        p.requires_grad = True

                # ===== Loss logic =====
                if epoch < 5:
                    # During warm-up, use random z (don't use encoder yet)
                    z = torch.randn(batch.size(0), 5, device=device)
                    t = torch.randint(0, ddpm.T, (batch.size(0),), device=device)
                    xt, noise = ddpm.noise_schedule(batch, t)
                    pred_noise = ddpm.model(xt, t, z)
                    loss_recon = F.mse_loss(pred_noise, noise)
                    loss_cls = torch.tensor(0.0, device=device)
                    loss_tc = torch.tensor(0.0, device=device)
                else:
                    # Normal training after warm-up
                    loss_recon, loss_cls, loss_tc = ddpm.train_step(batch, labels)

                if ddpm.classifier is not None:
                    z = ddpm.encoder(batch)
                    pred_factors = ddpm.classifier(z)
                    accs = []
                    for k in range(len(pred_factors)):
                        preds = pred_factors[k].argmax(dim=1)
                        acc = (preds == labels[:, k]).float().mean().item()
                        accs.append(acc)
                    avg_acc = sum(accs) / len(accs)
                    if i % 500 == 0:
                        print(f"Step {i} | Accuracies: {['%.3f' % a for a in accs]} | Avg: {avg_acc:.3f}")



            if torch.isnan(loss_recon): print("[NaN] loss_recon")
            if torch.isnan(loss_cls): print("[NaN] loss_cls")
            if torch.isnan(loss_tc): print("[NaN] loss_tc")

            cls_weight = 5.0
            total_loss = loss_recon + cls_weight * loss_cls + 0.1 * loss_tc

            if torch.isnan(total_loss):
                raise RuntimeError("[FATAL] NaN detected in total_loss")

            scaler.scale(total_loss).backward()


            # Clip gradients to avoid NaNs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # Update EMA (model and encoder)
            with torch.no_grad():
                for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                for p, ema_p in zip(encoder.parameters(), ema_encoder.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)


            if i % 100 == 0:
                with torch.no_grad():
                    t = torch.randint(0, ddpm.T, (batch.size(0),), device=batch.device)
                    xt, noise = ddpm.noise_schedule(batch, t)
                    mean_xt = xt.abs().mean().item()
                    mean_noise = noise.abs().mean().item()
                print(
                    f"Epoch {epoch} | Step {i} | "
                    f"Total: {total_loss:.4f} | Recon: {loss_recon:.4f} | Cls: {loss_cls:.4f} | TC: {loss_tc:.4f} | "
                    f"xt abs: {mean_xt:.4f} | noise abs: {mean_noise:.4f}"
                )

        # Save model and EMA samples
        if epoch % save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'ema_encoder_state_dict': ema_encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'epoch': epoch,
            }, f'./nopretraining_checkpoints/ddpm_epoch_{epoch}.pt')

            ddpm.model.eval()
            with torch.no_grad():
                real_batch, _ = next(iter(dataloader))
                real_batch = real_batch[:8].to(device)
                z = encoder(real_batch)
                t_zero = torch.zeros(8, dtype=torch.long, device=device)
                recon = ddpm.model(real_batch, t_zero, z)
                recon = (recon + 1) * 0.5
                save_image(recon, f'./nopretraining_samples/recon_t0_epoch_{epoch}.png', nrow=4)
            ddpm.model.train()


            print(f"Saved checkpoint at epoch {epoch}")

            # Debug sample 
            if epoch in [5, 10, 15]:  
                ddpm.model.eval()
                with torch.no_grad():
                    debug_sample = ddpm.sample(img_size=64, batch_size=8, z=None)
                debug_sample = (debug_sample + 1) * 0.5  # scale to [0, 1]
                save_image(debug_sample, f"./nopretraining_samples/debug_sample_epoch_{epoch}.png", nrow=4)
                print(f"[DEBUG] Saved debug_sample_epoch_{epoch}.png")
                ddpm.model.train()


            # Use EMA model temporarily for sampling
            ddpm.model = ema_model
            ddpm.encoder = ema_encoder 
            with torch.no_grad():
                sampled_imgs = ddpm.sample(img_size=64, batch_size=16)

            # Disentanglement visual check: latent traversals
            steps = 10
            traversals = latent_traversal(ddpm, z_dim=5, steps=steps, img_size=64)

            # Save full grid
            if not traversals:
                raise RuntimeError("Traversal list is empty — latent_traversal() failed.")
            print(f"Traversal shapes: {[t.shape for t in traversals]}")

            all_rows = torch.cat(traversals, dim=0)
            grid = make_grid(all_rows, nrow=steps, padding=2)
            save_image(grid, f'./nopretraining_samples/traversal_all_epoch_{epoch}.png')

            # Save per-dimension rows separately
            for i, imgs in enumerate(traversals):
                grid_i = make_grid(imgs, nrow=steps, padding=2)
                save_image(grid_i, f'./nopretraining_samples/traversal_z{i}_epoch_{epoch}.png')

            print(f"Saved full traversal grid for epoch {epoch}")


            # === Sample Quality Metrics ===
            with torch.no_grad():
                # Use a small fixed batch of real samples
                real_batch, _ = next(iter(dataloader))
                real_batch = real_batch[:16].to(device)
                gen_batch = ddpm.sample(img_size=64, batch_size=16).to(device)

                # L1 Distance (in [-1, 1] space)
                real_batch_up = F.interpolate(real_batch, size=64, mode="bilinear", align_corners=False)
                l1 = F.l1_loss(gen_batch * 2 - 1, real_batch_up).item()


                # Binariness score
                binariness = ((gen_batch < 0.05) | (gen_batch > 0.95)).float().mean().item()

                print(f"Sample Quality @ Epoch {epoch} | L1: {l1:.4f} | Binariness: {binariness:.4f}")

                # Save to text file
                with open("./nopretraining_samples/sample_quality.txt", "a") as f:
                    f.write(f"{epoch},{l1:.4f},{binariness:.4f}\n")
                with open("./nopretraining_samples/classifier_acc.txt", "a") as f:
                    acc_line = ",".join([f"{a:.4f}" for a in accs])
                    f.write(f"{epoch},{acc_line},{avg_acc:.4f}\n")


                pure_black = (sampled_imgs.view(sampled_imgs.size(0), -1).max(dim=1).values < 0.05).float().mean().item()
                print(f"Pure black ratio: {pure_black:.4f}")


            # Switch back to main model
            ddpm.model = model
            ddpm.encoder = encoder 


        scheduler.step(epoch + i / len(dataloader))  # fractional step 


