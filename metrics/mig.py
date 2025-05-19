"""Mutual Information Gap from the beta-TC-VAE paper. PyTorch version."""
import logging
import numpy as np
import gin
import metrics.utils as utils


@gin.configurable(
    "mig",
    denylist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_mig(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                batch_size=16):
  """Computes the mutual information gap."""
  del artifact_dir
  logging.info("Generating training set.")
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  return _compute_mig(mus_train, ys_train)



def _compute_mig(mus_train, ys_train):
  """Computes score based on both training and testing codes and factors."""
  score_dict = {}
  discretized_mus = utils.make_discretizer(mus_train)
  m = utils.discrete_mutual_info(discretized_mus, ys_train)
  #print("Mutual info matrix shape:", m.shape)
  #print("Mutual info matrix:", m)

  entropy = utils.discrete_entropy(ys_train)
  #print("Entropy:", entropy)


  # Filter out factors with zero entropy
  valid_factors = entropy > 1e-8
  if not np.any(valid_factors):
      score_dict["discrete_mig"] = float("nan")
      return score_dict

  sorted_m = np.sort(m, axis=0)[::-1]
  #print("Top 2 mutual infos per factor:", sorted_m[0, :], sorted_m[1, :])

  mig_per_factor = (sorted_m[0, valid_factors] - sorted_m[1, valid_factors]) / entropy[valid_factors]
  score_dict["discrete_mig"] = np.mean(mig_per_factor)
  return score_dict



@gin.configurable(
    "mig_validation",
    denylist=["observations", "labels", "representation_function"])
def compute_mig_on_fixed_data(observations, labels, representation_function,
                              batch_size=100):
  """Computes the MIG scores on the fixed set of observations and labels."""
  mus = utils.obtain_representation(observations, representation_function,
                                    batch_size)
  assert labels.shape[1] == observations.shape[0], "Wrong labels shape."
  assert mus.shape[1] == observations.shape[0], "Wrong representation shape."
  return _compute_mig(mus, labels)
