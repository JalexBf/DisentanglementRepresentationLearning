import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torchvision.utils as vutils

# Preprocessing function to normalize labels for conditioning
def preprocess_labels(labels):
    labels[:, 0:3] = labels[:, 0:3] / 0.90  # Normalize Floor, Wall, and Object Hue
    labels[:, 3] = (labels[:, 3] - 0.75) / (1.25 - 0.75)  # Normalize Object Size
    labels[:, 4:] = labels[:, 4:] / 1.0  # Normalize Shape and Orientation
    return torch.tensor(labels, dtype=torch.float32)

# Custom PyTorch Dataset for 3DShapes data
class Shapes3DDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.images = f['images'][:]
            labels = f['labels'][:]
        self.labels = preprocess_labels(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx] / 255.0  # Scale images to [0, 1]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), label

# 3-Layer U-Net for conditional noise prediction
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, cond_dim=6):
        super(UNet, self).__init__()

        self.label_embed = nn.Sequential(
            nn.Linear(cond_dim, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 8),
            nn.ReLU(),
            nn.Linear(base_channels * 8, base_channels * 4),
            nn.LayerNorm(base_channels * 4),
            nn.ReLU(),
        )

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + base_channels * 4, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.dec1 = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, cond):
        cond = (cond - cond.mean(dim=0, keepdim=True)) / (cond.std(dim=0, keepdim=True) + 1e-8)  # Normalize labels
        cond_embed = self.label_embed(cond).view(cond.size(0), -1, 1, 1)
        cond_embed = cond_embed.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_embed], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d3 = self.dec3(e3) + e2
        d2 = self.dec2(d3) + e1
        return self.dec1(d2)

# Linear Beta Schedule
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps, device="cuda")

# Forward Diffusion
def forward_diffusion(x_0, t, beta):
    batch_size = x_0.size(0)
    noise = torch.randn_like(x_0)
    beta_t = beta[t].view(batch_size, 1, 1, 1)
    alpha_bar_t = torch.cumprod(1 - beta, dim=0)[t].view(batch_size, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    print(f"Debug Forward Diffusion: x_0={x_0.mean().item()}, x_t={x_t.mean().item()}, noise={noise.mean().item()}")
    return x_t, noise

# Reverse Diffusion
def sample(model, beta, timesteps, x_t=None, cond=None):
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    if x_t is None:
        x_t = torch.randn((cond.size(0), 3, 64, 64), device="cuda")  # Initialize with noise

    for t in reversed(range(len(alpha_bar))):
        alpha_bar_t = alpha_bar[t].view(1, 1, 1, 1)
        noise_pred = model(x_t, cond)
        print(f"Debug Reverse Diffusion Step {t}: x_t_mean={x_t.mean().item()}, x_t_std={x_t.std().item()}, noise_pred_mean={noise_pred.mean().item()}, noise_pred_std={noise_pred.std().item()}")  # Debug output
        noise = torch.randn_like(x_t) if t > 0 else 0
        x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t) + torch.sqrt(beta[t]) * noise

    return x_t


# Loss Function
def diffusion_loss(model, x_0, t, cond, beta):
    x_t, noise = forward_diffusion(x_0, t, beta)
    pred_noise = model(x_t, cond)
    loss = F.mse_loss(pred_noise, noise)
    print(f"Debug Loss: Loss={loss.item()}")
    return loss

# Training Function
def train(model, dataloader, optimizer, scheduler, beta, timesteps, epochs, checkpoint_dir, sample_save_dir):
    os.makedirs(sample_save_dir, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.permute(0, 3, 1, 2).to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            t = torch.randint(0, timesteps, (images.size(0),), device="cuda").long()
            loss = diffusion_loss(model, images, t, labels, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

        # Save a sample
        with torch.no_grad():
            sample_noise = torch.randn((1, 3, 64, 64), device="cuda")
            labels_sample = labels[:1]
            generated_sample = sample(model, beta, timesteps, x_t=sample_noise, cond=labels_sample)
            vutils.save_image(generated_sample[0], f"{sample_save_dir}/sample_epoch_{epoch + 1}.png", normalize=True)

# Main Script
if __name__ == "__main__":
    dataset = Shapes3DDataset("3dshapes.h5")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = UNet(3, 3).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    timesteps = 100
    beta = linear_beta_schedule(timesteps)

    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        beta,
        timesteps,
        epochs=20,
        checkpoint_dir="./checkpoints",
        sample_save_dir="./samples"
    )
