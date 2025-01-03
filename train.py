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
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.Linear(base_channels * 4, base_channels * 8),
            nn.BatchNorm1d(base_channels * 8),
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
        # Normalize labels
        cond_std = torch.clamp(cond.std(dim=0, keepdim=True), min=1e-8)
        cond = (cond - cond.mean(dim=0, keepdim=True)) / cond_std

        # Embed and expand conditioning labels
        cond_embed = self.label_embed(cond).view(cond.size(0), -1, 1, 1)
        cond_embed = cond_embed.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond_embed], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Decoder
        d3 = self.dec3(e3) + e2
        d2 = self.dec2(d3) + e1
        output = torch.tanh(self.dec1(d2))  # Normalize output

        # Debugging intermediate outputs
        """print(f"Debug Encoder: e1_mean={e1.mean().item()}, e1_std={e1.std().item()}, "
              f"e2_mean={e2.mean().item()}, e2_std={e2.std().item()}, e3_mean={e3.mean().item()}, e3_std={e3.std().item()}")
        print(f"Debug Decoder: d3_mean={d3.mean().item()}, d3_std={d3.std().item()}, "
              f"d2_mean={d2.mean().item()}, d2_std={d2.std().item()}")"""

        return output


# Linear Beta Schedule
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.005):
    return torch.linspace(beta_start, beta_end, timesteps, device="cuda")


def forward_diffusion(x_0, t, beta):
    """
    Forward diffusion to compute x_t from x_0 for timestep t.
    """
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)  # Cumulative product of alphas

    # Select alpha_bar_t for the batch
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)

    # Generate noise
    noise = torch.randn_like(x_0)

    # Compute x_t
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    #print(f"Forward Diffusion: x_0 shape={x_0.shape}, x_t shape={x_t.shape}")

    return x_t, noise





# Reverse Diffusion
def sample(model, beta, timesteps, x_t, cond):
    """
    Reverse diffusion process to generate x_0 from x_T iteratively for a batch.
    """
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    batch_size = x_t.size(0)  # Ensure x_t has the batch dimension
    for t in reversed(range(timesteps)):
        alpha_t = alpha[t]
        alpha_bar_t = alpha_bar[t].view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)
        beta_t = beta[t].view(1, 1, 1, 1).expand(batch_size, -1, -1, -1)

        # Predict noise added at timestep t
        noise_pred = model(x_t, cond)

        # Compute mean of x_{t-1}
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
        )

        # Add noise if t > 0
        if t > 0:
            noise = torch.randn_like(x_t)  # Ensure batch-level noise
            variance = torch.sqrt(beta_t)
            x_t = mean + variance * noise
        else:
            x_t = mean  # No noise added at t=0


        """print(f"Reverse Diffusion: Initial x_t shape={x_t.shape}")

        print(f"Debug Reverse Diffusion Step {t}: x_t_mean={x_t.mean().item()}, x_t_std={x_t.std().item()}, "
              f"noise_pred_mean={noise_pred.mean().item()}, noise_pred_std={noise_pred.std().item()}")"""

    return x_t





# Loss Function
def diffusion_loss(model, x_0, t, cond, beta):
    """
    Compute the diffusion loss by comparing predicted noise with actual noise.
    """
    # Forward diffusion to get x_t and true noise
    x_t, noise = forward_diffusion(x_0, t, beta)

    # Predict noise using the model
    pred_noise = model(x_t, cond)

    # Compute mean squared error loss
    loss = 10.0 * F.mse_loss(pred_noise, noise)
    return loss



def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
    print(f"Checkpoint saved at {checkpoint_dir}/checkpoint_epoch_{epoch}.pth")


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



# Training Function
def train(model, dataloader, optimizer, scheduler, beta, timesteps, epochs, checkpoint_dir, sample_save_dir):
    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir)
    os.makedirs(sample_save_dir, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0
        for step, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            images = images.permute(0, 3, 1, 2).to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()

            # Random timestep for training
            t = torch.randint(0, timesteps, (images.size(0),), device="cuda").long()

            # Compute loss
            loss = diffusion_loss(model, images, t, labels, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Debug condition embedding stats every 100 steps
            if step % 100 == 0:
                with torch.no_grad():
                    cond_embed = model.label_embed(labels).view(labels.size(0), -1)
                    print(f"Epoch {epoch + 1}, Step {step}: Condition embedding mean={cond_embed.mean().item()}, std={cond_embed.std().item()}")

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

        save_checkpoint(model, optimizer, epoch, checkpoint_dir)

        # Generate and save a sample (batch-level)
        with torch.no_grad():
            batch_size = min(8, images.size(0))  # Ensure we don't exceed the current batch size
            x_t = torch.randn(batch_size, 3, 64, 64, device="cuda")  # Start from pure noise
            print(f"Sampling: Batch size={x_t.size(0)}, x_t shape={x_t.shape}")

            generated_sample = sample(model, beta, timesteps, x_t, labels[:batch_size])

            # Save only a few samples for visualization
            for i in range(batch_size):
                sample_save_path = os.path.join(sample_save_dir, f"sample_epoch_{epoch + 1}_{i}.png")
                vutils.save_image(generated_sample[i], sample_save_path, normalize=True)
                print(f"Sample saved to {sample_save_path}")




# Main Script
if __name__ == "__main__":
    dataset = Shapes3DDataset("3dshapes.h5")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = UNet(3, 3).to("cuda")
    optimizer = torch.optim.AdamW([
        {'params': model.label_embed.parameters(), 'lr': 1e-3},
        {'params': model.enc1.parameters()},
        {'params': model.enc2.parameters()},
        {'params': model.enc3.parameters()},
        {'params': model.dec3.parameters()},
        {'params': model.dec2.parameters()},
        {'params': model.dec1.parameters()}
    ], lr=5e-5)

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