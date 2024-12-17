import torch
import torch.nn.functional as F
import torchvision.utils as vutils
import h5py
import os
import gc
import argparse
from train import UNet, sample  # Import existing components


class Shapes3DDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.length = len(f['images'])  # Only load length metadata

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as f:
            image = f['images'][idx] / 255.0  # Normalize image
            label = f['labels'][idx]
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(label, dtype=torch.float32)


def generate_controlled_samples(model, beta, timesteps, base_label, factor_idx, save_dir, num_samples=10):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(num_samples):
            # Vary one factor while keeping others fixed
            new_label = base_label.clone().unsqueeze(0)
            new_label[0, factor_idx] = (i / (num_samples - 1))  # Linearly vary factor

            # Generate fresh noise and image
            x_t = torch.randn((1, 3, 64, 64), device="cpu")
            generated_sample = sample(model, beta, timesteps, x_t, new_label)

            # Save sample
            save_path = os.path.join(save_dir, f"sample_factor_{factor_idx}_step_{i}.png")
            vutils.save_image(generated_sample[0], save_path, normalize=True)
            print(f"Saved: {save_path}")

            # Memory cleanup for CPU
            del x_t, generated_sample, new_label
            gc.collect()


# Scheduler for cpu
def cosine_beta_schedule(timesteps, device="cpu"):  # Add device parameter
    return torch.tensor([
        0.008 * (1 - torch.cos(torch.tensor(i / timesteps * torch.pi / 2)))
        for i in range(timesteps)
    ], device=device)  # Set device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--factor_idx", type=int, required=True)
    args = parser.parse_args()

    # Load Dataset: Single Sample Loading
    dataset = Shapes3DDataset(args.dataset_path)
    base_image, base_label = dataset[0]  # Load only the first sample (for cpu)

    # Load Model
    model = UNet(3, 3, cond_dim=6).to("cpu")
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_epoch_9.pth")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")

    # Beta Schedule
    timesteps = 100
    beta = cosine_beta_schedule(timesteps, device="cpu")

    generate_controlled_samples(model, beta, timesteps, base_label, args.factor_idx, args.save_dir, num_samples=10)


if __name__ == "__main__":
    main()
