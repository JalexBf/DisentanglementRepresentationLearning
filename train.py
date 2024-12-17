import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import numpy as np
import os
import torchvision.utils as vutils
import argparse


# Preprocessing function to normalize labels for conditioning
def preprocess_labels(labels):
    # Normalize Floor, Wall, and Object Hue values to [0, 1]
    labels[:, 0:3] = labels[:, 0:3] / 0.90
    # Normalize Object Size between [-1, 1]
    labels[:, 3] = (labels[:, 3] - 0.75) / (1.25 - 0.75)
    # Normalize Shape and Orientation to [0, 1]
    labels[:, 4:] = labels[:, 4:] / 1.0
    return torch.tensor(labels, dtype=torch.float32)


# Custom PyTorch Dataset for 3DShapes data
class Shapes3DDataset(Dataset):
    def __init__(self, file_path):
        # Load images and labels from the HDF5 file into memory
        with h5py.File(file_path, 'r') as f:
            self.images = f['images'][:]
            labels = f['labels'][:]
        self.labels = preprocess_labels(labels)  # Normalize labels

    def __len__(self):
        return len(self.images)  # Total number of samples

    def __getitem__(self, idx):
        # Return normalized image and corresponding label
        image = self.images[idx] / 255.0  # Scale images to [0, 1]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), label


# 3-Layer U-Net for conditional noise prediction
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, cond_dim=6):
        super(UNet, self).__init__()

        # Conditioning labels embedding to match spatial feature size
        self.label_embed = nn.Sequential(
            nn.Linear(cond_dim, base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.ReLU(),
        )

        # Encoding (downsampling) layers
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels + base_channels * 4, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoding (upsampling) layers
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.dec1 = nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, cond):
        # Embed conditioning labels
        cond_embed = self.label_embed(cond)  # Shape: (B, base_channels * 4)
        cond_embed = cond_embed.view(cond_embed.size(0), cond_embed.size(1), 1, 1)  # Shape: (B, C, 1, 1)
        cond_embed = cond_embed.expand(-1, -1, x.size(2), x.size(3))  # Match input spatial dimensions (B, C, H, W)

        # Concatenate input image and conditioning embedding
        x = torch.cat([x, cond_embed], dim=1)  # Concatenate along channel dimension

        # Forward pass through U-Net with skip connections
        e1 = self.enc1(x)  # Downsample 1
        e2 = self.enc2(e1)  # Downsample 2
        e3 = self.enc3(e2)  # Downsample 3
        d3 = self.dec3(e3) + e2  # Upsample 1 + skip connection
        d2 = self.dec2(d3) + e1  # Upsample 2 + skip connection
        d1 = self.dec1(d2)  # Final output
        return d1


# Cosine Beta Schedule: Generates noise variance across timesteps
def cosine_beta_schedule(timesteps):
    return torch.tensor([
        0.008 * (1 - torch.cos(torch.tensor(i / timesteps * torch.pi / 2)))
        for i in range(timesteps)
    ], device="cuda")


# Forward diffusion process: Add noise to images
def forward_diffusion(x_0, t, beta):
    """
    Add Gaussian noise to an image at a specific timestep t.
    Args:
        x_0 (Tensor): Original clean image, shape (B, C, H, W).
        t (Tensor): Timesteps, shape (B,).
        beta (Tensor): Beta schedule, shape (timesteps,).
    Returns:
        x_t (Tensor): Noisy image at timestep t.
        noise (Tensor): Added noise.
    """
    batch_size = x_0.size(0)
    noise = torch.randn_like(x_0)  # Generate Gaussian noise
    beta_t = beta[t].view(batch_size, 1, 1, 1)  # Select beta for each timestep in batch
    alpha_bar_t = torch.cumprod(1 - beta, dim=0)[t].view(batch_size, 1, 1, 1)

    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise





# Reverse diffusion: Sample clean image from noise
def sample(model, beta, timesteps, x_t, cond):
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    for t in reversed(range(len(alpha_bar))):
        alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
        noise_pred = model(x_t, cond)  # Predict noise

        # Add noise except for the last timestep
        noise = torch.randn_like(x_t) if t > 0 else 0
        x_t = (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t) + torch.sqrt(beta[t]) * noise

    return x_t


# Loss function: Weighted MSE between predicted and actual noise
def diffusion_loss(model, x_0, t, cond, beta):
    """
    Compute the MSE loss for diffusion noise prediction.
    Args:
        model: U-Net model.
        x_0 (Tensor): Clean input image.
        t (Tensor): Timesteps.
        cond (Tensor): Conditioning labels.
        beta (Tensor): Beta schedule.
    Returns:
        loss (Tensor): Loss value.
    """
    x_t, noise = forward_diffusion(x_0, t, beta)  # Add noise at timestep t
    pred_noise = model(x_t, cond)  # Predict noise
    loss = F.mse_loss(pred_noise, noise)  # MSE loss
    return loss



def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}.")


def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")]
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}. Starting from epoch 0.")
        return 0
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {checkpoint['epoch'] + 1}.")
    return checkpoint['epoch'] + 1


def train(model, dataloader, optimizer, scheduler, beta, timesteps, epochs, checkpoint_dir, sample_save_dir):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir)
    os.makedirs(sample_save_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.permute(0, 3, 1, 2).to("cuda")  # Change to NCHW format
            labels = labels.to("cuda")  # Move labels to GPU

            optimizer.zero_grad()

            # Randomly choose timesteps t for each image in the batch
            t = torch.randint(0, timesteps, (images.size(0),), device="cuda").long()

            # Compute loss using forward diffusion and noise prediction
            loss = diffusion_loss(model, images, t, labels, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

        # Generate sample images
        with torch.no_grad():
            sample_noise = torch.randn((1, 3, 64, 64), device="cuda")
            labels_sample = labels[:1]
            generated_sample = sample(model, beta, timesteps, sample_noise, labels_sample)
            sample_save_path = os.path.join(sample_save_dir, f"sample_epoch_{epoch + 1}.png")
            vutils.save_image(generated_sample[0], sample_save_path, normalize=True)
            print(f"Sample saved to {sample_save_path}")






# Main script
if __name__ == "__main__":
    dataset = Shapes3DDataset("3dshapes.h5")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = UNet(3, 3).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    timesteps = 100
    beta = cosine_beta_schedule(timesteps)

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
