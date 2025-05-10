import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from classifier_guided import UNet, DSpritesLazyDataset
from metrics import evaluate_metrics  # Assuming metrics.py contains the updated MIG, SAP, DCI code
from model import BetaVAE_H, BetaVAE_B

# ------------------------------
# Configuration
# ------------------------------
CHECKPOINT_PATH = './classifier_checkpoints/ddpm_epoch_20.pt'
DATASET_PATH = './data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
BATCH_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------
# Load Model
# ------------------------------

def load_model(checkpoint_path, model_type, z_dim, nc, device):
    if model_type == 'H':
        model = BetaVAE_H(z_dim=z_dim, nc=nc).to(device)
    elif model_type == 'B':
        model = BetaVAE_B(z_dim=z_dim, nc=nc).to(device)
    else:
        raise ValueError("Invalid model type. Choose 'H' or 'B'.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_states"]["net"])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model


# ------------------------------
# Extract Latents
# ------------------------------
def extract_latents(model, dataloader, device):
    latents = []
    labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, targets) in enumerate(dataloader):
            print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            print(f"targets shape: {targets.shape}")
            print(f"First target sample: {targets[0]}")

            imgs, targets = imgs.to(device), targets.to(device)

            # Zero out the color factor
            targets[:, 0] = 0

            # Extract conditional embeddings
            try:
                latent_batch = model.cond_emb(targets)
                latents.append(latent_batch.cpu())
                labels.append(targets.cpu())
            except Exception as e:
                print(f"Error during cond_emb processing: {e}")
                break

        # If no data was processed
        if not latents:
            print("No latents extracted. Exiting...")
            return None, None

        print(f"Collected {len(latents)} batches of latents and labels")
        print(f"Latent shape of first batch: {latents[0].shape}")
        print(f"Label shape of first batch: {labels[0].shape}")

        try:
            # Concatenate latents and labels
            latents = torch.cat(latents, dim=0).numpy()
            labels = torch.cat(labels, dim=0).numpy()

            print(f"Final latents shape: {latents.shape}")
            print(f"Final labels shape: {labels.shape}")
        except Exception as e:
            print(f"Error during concatenation: {e}")
            return None, None

        # Standardize latents
        scaler = StandardScaler()
        latents = scaler.fit_transform(latents)

        return latents, labels



# ------------------------------
# Main Function
# ------------------------------
def main():
    # Load model
    model = load_model(CHECKPOINT_PATH, DEVICE)

    # Load dataset
    dataset = DSpritesLazyDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Extract latents and labels
    latents, labels = extract_latents(model, dataloader, DEVICE)

    # Evaluate metrics
    scores = evaluate_metrics(torch.tensor(latents), torch.tensor(labels))
    print(f"MIG: {scores['MIG']:.4f}")
    print(f"SAP: {scores['SAP']:.4f}")
    print(f"DCI: {scores['DCI']}")

if __name__ == "__main__":
    main()
