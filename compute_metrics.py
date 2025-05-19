import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress TensorFlow logs

import numpy as np
import gin
import time
from disentanglement_lib.evaluation.metrics import mig, dci, sap_score


class NumpyGroundTruth:
    def __init__(self, factors):
        self.factors = factors
        self.num_total_factors = factors.shape[1]

    def sample(self, num_points, random_state):
        indices = random_state.choice(self.factors.shape[0], num_points)
        return None, self.factors[indices]  # MIG only uses factors



# Bind only MIG gin parameter
gin.bind_parameter("mig.num_train", 1000)

# Load data
data = np.load("representation.npz")
z = data["z"]
factors = data["y"]

# Set random seed
random_state = np.random.RandomState(42)

start = time.time()

class RepresentationFunction:
    def __init__(self, z):
        self.z = z
        self.index = 0

    def __call__(self, x):
        batch_size = x.shape[0]
        result = self.z[self.index:self.index + batch_size]
        self.index += batch_size
        return result


# MIG needs random_state
ground_truth = NumpyGroundTruth(factors)
rep_fn = RepresentationFunction(z)
mig_score = mig.compute_mig(
    ground_truth,
    rep_fn,
    random_state=random_state
)["mig"]

print(f"MIG: {mig_score:.4f} (took {time.time() - start:.2f}s)")

# DCI and SAP are fine
dci_scores = dci.compute_dci(z, factors, train_size=10000, test_size=5000)
sap = sap_score.compute_sap(z, factors)["sap_score"]

# Output results
print(f"DCI: disentanglement={dci_scores['disentanglement']:.4f}, completeness={dci_scores['completeness']:.4f}")
print(f"SAP: {sap:.4f}")
