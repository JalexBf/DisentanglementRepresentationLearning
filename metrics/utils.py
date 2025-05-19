# coding=utf-8
"""Utility functions for disentanglement metrics. PyTorch version."""
import numpy as np
import torch
from sklearn.metrics import mutual_info_score
import gin


def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size):
  """Sample a single training sample based on a mini-batch of ground-truth data."""
  representations = None
  factors = None
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_factors, current_observations = \
        ground_truth_data.sample(num_points_iter, random_state)
    current_observations = torch.from_numpy(current_observations).float()
    current_representations = representation_function(current_observations).detach().cpu().numpy()
    if i == 0:
      factors = current_factors
      representations = current_representations
    else:
      factors = np.vstack((factors, current_factors))
      representations = np.vstack((representations, current_representations))
    i += num_points_iter
  return np.transpose(representations), np.transpose(factors)


def obtain_representation(observations, representation_function, batch_size):
  """Obtain representations from observations."""
  representations = None
  num_points = observations.shape[0]
  i = 0
  while i < num_points:
    num_points_iter = min(num_points - i, batch_size)
    current_observations = torch.from_numpy(observations[i:i + num_points_iter]).float()
    current_representations = representation_function(current_observations).detach().cpu().numpy()
    if i == 0:
      representations = current_representations
    else:
      representations = np.vstack((representations, current_representations))
    i += num_points_iter
  return np.transpose(representations)


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
  return m


def discrete_entropy(ys):
  """Compute discrete entropy."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = mutual_info_score(ys[j, :], ys[j, :])
  return h


@gin.configurable(
    "discretizer", denylist=["target"])
def make_discretizer(target, num_bins=gin.REQUIRED,
                     discretizer_fn=gin.REQUIRED):
  """Wrapper that creates discretizers."""
  return discretizer_fn(target, num_bins)


@gin.configurable("histogram_discretizer", denylist=["target"])
def _histogram_discretize(target, num_bins=gin.REQUIRED):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized
