import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import gc


def compute_entropy(values):
    """Compute entropy of a discrete distribution."""
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


def compute_mutual_info(latent, factor):
    """Compute mutual information between a latent dimension and a factor."""
    mi = mutual_info_score(factor, latent)
    return mi


def discretize_latents(latents, num_bins=10):
    """Discretize continuous latent values into bins."""
    latents_disc = np.zeros_like(latents)
    for i in range(latents.shape[1]):
        latents_disc[:, i] = np.digitize(latents[:, i], np.histogram(latents[:, i], bins=num_bins)[1][:-1])
    return latents_disc


def compute_mig(latents, labels, num_bins=10):
    """
    Compute Mutual Information Gap (MIG).

    Args:
        latents (np.ndarray): Latent variables with shape (N, D).
        labels (np.ndarray): Ground truth factors with shape (N, K).
        num_bins (int): Number of bins for discretizing latents.

    Returns:
        float: MIG score.
    """
    # Discretize latents
    latents_disc = discretize_latents(latents, num_bins=num_bins)

    num_latents = latents_disc.shape[1]
    num_factors = labels.shape[1]

    mig_scores = []

    for i in range(num_factors):
        factor_values = labels[:, i]
        mi_scores = []

        for j in range(num_latents):
            latent_values = latents_disc[:, j]
            mi = mutual_info_score(factor_values, latent_values)
            mi_scores.append(mi)

        mi_scores = np.array(mi_scores)
        sorted_mi = np.sort(mi_scores)[::-1]

        # Calculate entropy of the factor
        entropy = compute_entropy(factor_values)

        if entropy > 0 and len(sorted_mi) > 1:
            mig = (sorted_mi[0] - sorted_mi[1]) / entropy
            mig_scores.append(mig)

    return np.mean(mig_scores) if mig_scores else 0.0



def compute_sap(latents, labels):
    """
    Compute Separate Attribute Predictability (SAP).

    Args:
        latents (np.ndarray): Latent variables with shape (N, D).
        labels (np.ndarray): Ground truth factors with shape (N, K).

    Returns:
        float: SAP score.
    """
    num_latents = latents.shape[1]
    num_factors = labels.shape[1]

    sap_scores = []

    for i in range(num_factors):
        factor_values = labels[:, i]

        accuracies = []
        for j in range(num_latents):
            latent_values = latents[:, j].reshape(-1, 1)

            try:
                clf = LogisticRegression(max_iter=500)
                clf.fit(latent_values, factor_values)
                preds = clf.predict(latent_values)
                acc = accuracy_score(factor_values, preds)
                accuracies.append(acc)
            except:
                accuracies.append(0.0)

        accuracies = np.array(accuracies)
        sorted_accuracies = np.sort(accuracies)[::-1]

        if len(sorted_accuracies) > 1:
            sap = sorted_accuracies[0] - sorted_accuracies[1]
            sap_scores.append(sap)

    return np.mean(sap_scores) if sap_scores else 0.0


def compute_dci(latents, labels):
    """
    Compute Disentanglement, Completeness, and Informativeness (DCI).

    Args:
        latents (np.ndarray): Latent variables with shape (N, D).
        labels (np.ndarray): Ground truth factors with shape (N, K).

    Returns:
        dict: DCI scores.
    """
    num_latents = latents.shape[1]
    num_factors = labels.shape[1]

    importance_matrix = np.zeros((num_factors, num_latents))

    for i in range(num_factors):
        factor_values = labels[:, i]

        for j in range(num_latents):
            latent_values = latents[:, j].reshape(-1, 1)

            try:
                clf = LogisticRegression(max_iter=500)
                clf.fit(latent_values, factor_values)
                importance_matrix[i, j] = clf.score(latent_values, factor_values)
            except:
                importance_matrix[i, j] = 0.0

    # Disentanglement
    disentanglement_scores = 1 - importance_matrix.var(axis=0) / (importance_matrix.var() + 1e-10)
    disentanglement = disentanglement_scores.mean()

    # Completeness
    completeness_scores = 1 - importance_matrix.var(axis=1) / (importance_matrix.var() + 1e-10)
    completeness = completeness_scores.mean()

    # Informativeness
    informativeness = importance_matrix.mean()

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": informativeness
    }



import gc

def evaluate_metrics(latents, labels, sample_size=50000):
    """
    Evaluate all metrics (MIG, SAP, DCI) with optional sampling.

    Args:
        latents (torch.Tensor): Latent variables (N, D).
        labels (torch.Tensor): Ground truth factors (N, K).
        sample_size (int): Number of samples to use for metrics calculation.

    Returns:
        dict: MIG, SAP, DCI scores.
    """
    print(f"Latents shape: {latents.shape}")
    print(f"Labels shape: {labels.shape}")

    latents = latents.cpu().numpy()
    labels = labels.cpu().numpy()

    # Standardize latents
    scaler = StandardScaler()
    latents = scaler.fit_transform(latents)

    print(f"Standardized latents shape: {latents.shape}")

    # Sampling to avoid memory overflow
    if latents.shape[0] > sample_size:
        indices = np.random.choice(latents.shape[0], sample_size, replace=False)
        latents = latents[indices]
        labels = labels[indices]
        print(f"Reduced to {sample_size} samples for metrics calculation.")

    # Inspect a few samples
    print("First 5 latent samples:\n", latents[:5])
    print("First 5 label samples:\n", labels[:5])

    # Explicitly release memory
    del scaler
    gc.collect()

    # Calculate metrics
    try:
        print("Calculating MIG...")
        mig_score = compute_mig(latents, labels)
        print(f"MIG calculated: {mig_score}")
    except Exception as e:
        print(f"Error in MIG calculation: {e}")
        mig_score = None

    try:
        print("Calculating SAP...")
        sap_score = compute_sap(latents, labels)
        print(f"SAP calculated: {sap_score}")
    except Exception as e:
        print(f"Error in SAP calculation: {e}")
        sap_score = None

    try:
        print("Calculating DCI...")
        dci_scores = compute_dci(latents, labels)
        print(f"DCI calculated: {dci_scores}")
    except Exception as e:
        print(f"Error in DCI calculation: {e}")
        dci_scores = None

    # Additional memory cleanup
    gc.collect()

    return {
        "MIG": mig_score,
        "SAP": sap_score,
        "DCI": dci_scores
    }

