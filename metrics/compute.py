import torch
import numpy as np
from torch.utils.data import Dataset
from metrics.mig import compute_mig
from metrics.dci import compute_dci
from metrics.sap import compute_sap
import metrics.utils as utils
from unsupervised import LatentEncoder  
from unsupervised import DSpritesLazyDataset  
import gin


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = LatentEncoder(z_dim=10).to(device)
encoder.load_state_dict(torch.load('unsupervised_checkpoints/encoder_epoch_20.pt', map_location=device))
encoder.eval()


# -------------------------
# Define representation_function
# -------------------------
def representation_function(x_batch):
    x_batch = x_batch.to(device)
    with torch.no_grad():
        z = encoder(x_batch)
    return z


# -------------------------
# Wrap dSprites Dataset
# -------------------------
class TorchDSpritesWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def sample(self, num_points, random_state):
        indices = random_state.choice(len(self.dataset), size=num_points, replace=False)
        imgs = []
        latents = []
        for idx in indices:
            img, latent = self.dataset[idx]
            imgs.append(img.numpy())         # shape: (1, 64, 64)
            latents.append(latent.numpy())   # shape: (6,)
        imgs = np.stack(imgs)                # shape: (N, 1, 64, 64)
        latents = np.stack(latents)          # shape: (N, 6)
        return latents, imgs


# -------------------------
# Main evaluation
# -------------------------
if __name__ == "__main__":
    dataset = DSpritesLazyDataset("./dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    wrapped_data = TorchDSpritesWrapper(dataset)
    random_state = np.random.RandomState(42)

    # -------------------------
    # MIG
    # -------------------------
    gin.bind_parameter("discretizer.num_bins", 20)
    gin.bind_parameter("discretizer.discretizer_fn", utils._histogram_discretize)

    mig_result = compute_mig(
        ground_truth_data=wrapped_data,
        representation_function=representation_function,
        random_state=random_state,
        num_train=10000,
        batch_size=64
    )
    print("MIG score:", mig_result["discrete_mig"])

    # -------------------------
    # DCI
    # -------------------------
    gin.bind_parameter("dci.num_train", 10000)
    gin.bind_parameter("dci.num_test", 5000)

    dci_result = compute_dci(
        ground_truth_data=wrapped_data,
        representation_function=representation_function,
        random_state=random_state,
        num_train=10000,
        num_test=5000,
        batch_size=64
    )

    print("DCI disentanglement:", dci_result["disentanglement"])
    print("DCI completeness:", dci_result["completeness"])
    print("DCI informativeness (train):", dci_result["informativeness_train"])
    print("DCI informativeness (test):", dci_result["informativeness_test"])

   
    # -------------------------
    # SAP
    # -------------------------
    gin.bind_parameter("sap_score.num_train", 10000)
    gin.bind_parameter("sap_score.num_test", 5000)
    gin.bind_parameter("sap_score.continuous_factors", False)

    sap_result = compute_sap(
        ground_truth_data=wrapped_data,
        representation_function=representation_function,
        random_state=random_state,
        num_train=10000,
        num_test=5000,
        batch_size=64,
        continuous_factors=False
    )

    print("SAP score:", sap_result["SAP_score"])

