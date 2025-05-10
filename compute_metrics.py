import numpy as np
from disentanglement_lib.evaluation.metrics import mig, dci, sap_score


class DataWrapper:
    """A wrapper to provide a `.sample()` method for the factors array."""

    def __init__(self, factors):
        self.factors = factors

    def sample(self, num_samples, random_state):
        """Returns a batch of samples with factors and indices."""
        rng = np.random.default_rng(random_state)
        indices = rng.integers(0, len(self.factors), size=num_samples)
        sampled_factors = self.factors[indices]
        return sampled_factors, indices


def representation_function(latents, indices):
    """A simple representation function that returns the latents given indices."""
    return latents[indices]


def compute_disentanglement_metrics(latents_path, factors_path, num_train=10000, random_state=42):
    """
    Computes MIG, DCI, SAP metrics using the Disentanglement Library.
    """
    latents = np.load(latents_path)
    factors = np.load(factors_path)

    # Wrap factors in DataWrapper to provide a `.sample()` method for MIG only
    ground_truth_data = DataWrapper(factors)

    # MIG
    mig_score = mig.compute_mig(
        ground_truth_data=ground_truth_data,
        representation_function=lambda indices: representation_function(latents, indices),
        num_train=num_train,
        random_state=random_state
    )["discrete_mig"]

    # DCI (uses the original `factors` directly)
    dci_scores = dci.compute_dci(latents, factors)
    dci_disentanglement = dci_scores["disentanglement"]
    dci_completeness = dci_scores["completeness"]
    dci_informativeness = dci_scores["informativeness_test"]

    # SAP (uses the original `factors` directly)
    sap_score_value = sap_score.compute_sap(latents, factors)["SAP_score"]

    metrics = {
        "MIG": mig_score,
        "DCI Disentanglement": dci_disentanglement,
        "DCI Completeness": dci_completeness,
        "DCI Informativeness": dci_informativeness,
        "SAP": sap_score_value,
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    latents_path = "latents.npy"
    factors_path = "factors.npy"
    
    # Adjust `num_train` as needed
    compute_disentanglement_metrics(latents_path, factors_path, num_train=10000, random_state=42)
