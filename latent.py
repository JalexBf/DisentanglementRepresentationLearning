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
from stupid_encoder import LatentEncoder, LatentClassifier 

torch.manual_seed(0)
np.random.seed(0)


# ------------------------------
# Dataset Loader
# ------------------------------
class DSpritesLazyDataset(Dataset):
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
        self.imgs = self.data['imgs']
        self.latents_classes = self.data['latents_classes']  # Use class indices instead of continuous

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        img = (torch.from_numpy(img) > 0.5).float().unsqueeze(0) * 2.0 - 1.0
        labels = torch.floor(torch.from_numpy(self.latents_classes[idx])).long()
        labels = torch.clamp(labels, max=31)  # optional: avoid overflow just in case
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
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_dim):
        super().__init__()
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.cond_preprocess = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels)
        )


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

    def forward(self, x, t, z):
        h = self.block1(x)
        t_emb = self.time_emb_proj(t).unsqueeze(-1).unsqueeze(-1)
        z_emb = self.cond_preprocess(z).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb + z_emb
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
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128, cond_dim=10):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Channels
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

    def forward(self, x, t, z):
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
    def __init__(self, model, encoder=None, classifier=None, timesteps=500, beta_start=1e-4, beta_end=0.01):
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

    @torch.no_grad()
    def get_z(self, x):
        return self.encoder(x)

    def train_step(self, x0, gt):
        self.model.train()
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), dtype=torch.long, device=device)

        # Use fixed pretrained encoder
        with torch.no_grad():
            z = self.encoder(x0)

        xt, noise = self.noise_schedule(x0, t)
        pred_noise = self.model(xt, t, z)

        loss_recon = F.mse_loss(pred_noise, noise)
        pred_factors = self.classifier(z)
        loss_cls = sum(F.cross_entropy(pred_factors[i], gt[:, i]) for i in range(5))

        loss = loss_recon + 5.0 * loss_cls
        return loss, loss_recon, loss_cls




    @torch.no_grad()
    def sample(self, img_size, batch_size=256, z=None, model_override=None):
        model = model_override or self.model
        device = next(self.model.parameters()).device
        img = torch.randn(batch_size, 1, img_size, img_size, device=device)

        if z is None:
            z = torch.randn(batch_size, self.encoder.encoder[-1].out_features, device=device)



        for t in reversed(range(self.T)):
            time = torch.full((batch_size,), t, device=device, dtype=torch.long)
            pred_noise = model(img, time, z)

            alpha = self.alpha[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.beta[t]

            if t > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (1 / alpha.sqrt()) * (img - ((1 - alpha) / (1 - alpha_hat).sqrt()) * pred_noise) + beta.sqrt() * noise

        return img.clamp(-1, 1)




@torch.no_grad()
def latent_traversal(ddpm, z_dim=11, steps=10, img_size=64, fixed_z=None):
    device = next(ddpm.model.parameters()).device
    if fixed_z is None:
        fixed_z = torch.randn(1, z_dim, device=device)

    all_imgs = []

    for i in range(z_dim):
        z_batch = fixed_z.repeat(steps, 1)
        linspace = torch.linspace(-3, 3, steps).to(device)
        z_batch[:, i] = linspace  # vary only dim i

        imgs = ddpm.sample(img_size=img_size, batch_size=steps, z=z_batch)
        imgs = (imgs + 1) * 0.5  # scale to [0, 1]
        all_imgs.append(imgs)

    return all_imgs



def classification_accuracies(preds, labels):
    accs = []
    for i in [0, 1, 2, 3, 4]:  
        pred_classes = preds[i].argmax(dim=1)
        true_classes = labels[:, i].long()
        acc = (pred_classes == true_classes).float().mean().item()
        accs.append(acc)
    return accs  
# ------------------------------
# Training Loop
# ------------------------------
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DSpritesLazyDataset('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=4)

    model = UNet(base_channels=64, cond_dim=11).to(device)
    ema_model = deepcopy(model)  

    ema_decay = 0.995

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    encoder = LatentEncoder(z_dim=11).to(device)
    classifier = LatentClassifier().to(device)

    checkpoint_dir = './latent_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)  
    latest_ckpt = sorted(
        [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    checkpoint = None
    start_epoch = 1

    if latest_ckpt:
        path = os.path.join(checkpoint_dir, latest_ckpt[-1])
        print(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if 'ema_state_dict' in checkpoint:
            ema_state = checkpoint['ema_state_dict']
            ema_state = {k.replace('_orig_mod.', ''): v for k, v in ema_state.items()}
            ema_model.load_state_dict(ema_state)

        else:
            print("No EMA state found in checkpoint. Reinitializing EMA.")

    # Always load pretrained encoder/classifier AFTER checkpoint
    pretrained = torch.load('pretrained.pt', map_location=device)
    encoder.load_state_dict(pretrained['encoder'])
    classifier.load_state_dict(pretrained['classifier'])
    print("Loaded pretrained encoder/classifier.")

    for p in encoder.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = False

    print("Classifier heads:")
    for name, layer in classifier.named_children():
        print(f" - {name}: {layer.out_features} classes")


    # === Sanity check ===
    with torch.no_grad():
        batch, labels = next(iter(dataloader))
        batch = batch.to(device)
        labels = labels[:, 1:6].clone().to(device)



    print("Classes per head:", [
        classifier.shape_head.out_features,
        classifier.scale_head.out_features,
        classifier.orient_head.out_features,
        classifier.pos_x_head.out_features,
        classifier.pos_y_head.out_features,
    ])

    encoder.eval()
    classifier.eval()


    ddpm = DDPM(model, encoder=encoder, classifier=classifier)

    # Freeze encoder and classifier parameters
    for p in encoder.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = False

    os.makedirs('./latent_checkpoints', exist_ok=True)
    os.makedirs('./latent_samples', exist_ok=True)

    scaler = torch.cuda.amp.GradScaler()
    save_every = 1

    for epoch in range(start_epoch, 100):
        for i, (batch, labels) in enumerate(dataloader):
            batch = batch.to(device)
            labels_full = labels.to(device)
            labels_cls = labels_full.clone().long()
            assert labels_cls.min() >= 0
            assert labels_cls[:, 0].max() < 3   # shape
            assert labels_cls[:, 1].max() < 6   # scale
            assert labels_cls[:, 2].max() < 40  # orientation
            assert labels_cls[:, 3].max() < 32  # pos_x
            assert labels_cls[:, 4].max() < 32  # pos_y


            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss, loss_recon, loss_cls = ddpm.train_step(batch, labels_cls)

            scaler.scale(loss).backward()

            # Clip gradients to avoid NaNs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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
                print(f"Epoch {epoch} | Step {i} | Total: {loss.item():.4f} | Recon: {loss_recon.item():.4f} | Cls: {loss_cls:.4f}")
                print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
                with torch.no_grad():
                    z = encoder(batch)
                    pred_factors = classifier(z)
                    assert len(pred_factors) == 5, "Classifier output does not have 5 heads"
                    accs = classification_accuracies(pred_factors, labels_cls)
                    assert labels_cls.shape[1] == 5, f"Expected 5 label columns, got {labels_cls.shape[1]}"
                    print(f"Accuracies | Shape: {accs[0]:.3f} | Scale: {accs[1]:.3f} | Rotation: {accs[2]:.3f} | PosX: {accs[3]:.3f} | PosY: {accs[4]:.3f}")




        # Save model and EMA samples
        if epoch % save_every == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),  
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'encoder_state_dict': encoder.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'epoch': epoch,
            }, f'./latent_checkpoints/ddpm_epoch_{epoch}.pt')

            ddpm.model.eval()
            with torch.no_grad():
                real_batch, _ = next(iter(dataloader))
                real_batch = real_batch[:8].to(device)
                z = encoder(real_batch)
                t_zero = torch.zeros(8, dtype=torch.long, device=device)
                recon = ddpm.model(real_batch, t_zero, z)
                recon = (recon + 1) * 0.5
                save_image(recon, f'./latent_samples/recon_t0_epoch_{epoch}.png', nrow=4)
            ddpm.model.train()


            print(f"Saved checkpoint at epoch {epoch}")

            # Use EMA model temporarily for sampling
            ddpm.model = ema_model
            with torch.no_grad():
                sampled_imgs = ddpm.sample(img_size=64, batch_size=11, model_override=ema_model)

            # Disentanglement visual check: latent traversals
            if epoch == start_epoch:
                fixed_z = torch.randn(1, encoder.encoder[-1].out_features, device=device)
            traversals = latent_traversal(ddpm, steps=10, img_size=64, fixed_z=fixed_z)


            all_rows = torch.cat(traversals, dim=0)  # shape: [z_dim * steps, 1, 64, 64]
            grid = make_grid(all_rows, nrow=traversals[0].size(0), padding=2)
            save_image(grid, f'./latent_samples/traversal_all_epoch_{epoch}.png')
            print(f"Saved full traversal grid for epoch {epoch}")


            # === Sample Quality Metrics ===
            with torch.no_grad():
                # Use a small fixed batch of real samples
                real_batch, _ = next(iter(dataloader))
                real_batch = real_batch[:16].to(device)
                gen_batch = ddpm.sample(img_size=64, batch_size=16).to(device)

                # L1 Distance (in [-1, 1] space)
                l1 = F.l1_loss(gen_batch * 2 - 1, real_batch).item()

                # Binariness score
                binariness = ((gen_batch < 0.05) | (gen_batch > 0.95)).float().mean().item()

                print(f"Sample Quality @ Epoch {epoch} | L1: {l1:.4f} | Binariness: {binariness:.4f}")

                # Save to text file
                with open("./latent_samples/sample_quality.txt", "a") as f:
                    f.write(f"{epoch},{l1:.4f},{binariness:.4f}\n")

                pure_black = (sampled_imgs.view(sampled_imgs.size(0), -1).max(dim=1).values < 0.05).float().mean().item()
                print(f"Pure black ratio: {pure_black:.4f}")


            # Switch back to main model
            ddpm.model = model



        scheduler.step()