import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from classifier_guided import UNet, DSpritesLazyDataset  

def extract_latents_and_factors(npz_path, model_path, output_latents_path, output_factors_path, batch_size=64):
    """
    Extracts the conditional embeddings (as latents) and ground truth factors from the trained model.
    Saves them as .npy files for metric evaluation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = DSpritesLazyDataset(npz_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load model
    model = UNet(base_channels=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_latents = []
    all_factors = []

    for imgs, latents in tqdm(dataloader, desc="Extracting Latents"):
        latents = latents.to(device)

        # Extract conditional embeddings as "latent representations"
        with torch.no_grad():
            cond_emb = model.cond_emb(latents)  # Shape: (B, emb_dim)
            all_latents.append(cond_emb.cpu().numpy())
            all_factors.append(latents[:, 1:].cpu().numpy())  # Exclude dataset index

    # Concatenate all batches
    all_latents = np.concatenate(all_latents, axis=0)
    all_factors = np.concatenate(all_factors, axis=0)

    # Save as .npy files
    np.save(output_latents_path, all_latents)
    np.save(output_factors_path, all_factors)

    print(f"Saved latents to {output_latents_path}")
    print(f"Saved factors to {output_factors_path}")


if __name__ == "__main__":
    dataset_path = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    model_checkpoint = "./classifier_checkpoints/ddpm_epoch_20.pt"

    # Output paths for latents and factors
    output_latents_path = "latents.npy"
    output_factors_path = "factors.npy"

    extract_latents_and_factors(dataset_path, model_checkpoint, output_latents_path, output_factors_path)
