"""Implementation of the SAP score (PyTorch version)."""
import numpy as np
from sklearn import svm
import gin
import metrics.utils as utils


@gin.configurable(
    "sap_score",
    denylist=["ground_truth_data", "representation_function", "random_state", "artifact_dir"])
def compute_sap(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=64,
                continuous_factors=gin.REQUIRED):
    del artifact_dir

    mus, ys = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train,
        random_state, batch_size)

    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test,
        random_state, batch_size)

    return _compute_sap(mus, ys, mus_test, ys_test, continuous_factors)


def _compute_sap(mus, ys, mus_test, ys_test, continuous_factors):
    score_matrix = compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors)
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]

    return {"SAP_score": compute_avg_diff_top_two(score_matrix)}


def compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros((num_latents, num_factors))

    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]

            if continuous_factors:
                cov = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov[0, 1] ** 2
                var_mu = cov[0, 0]
                var_y = cov[1, 1]
                score_matrix[i, j] = cov_mu_y / (var_mu * var_y) if var_mu > 1e-12 else 0.
            else:
                mu_i_test = mus_test[i, :]
                y_j_test = ys_test[j, :]

                # Only use if >1 class exists
                if len(np.unique(y_j)) < 2:
                    continue

                try:
                    clf = svm.LinearSVC(C=0.01, class_weight="balanced", max_iter=1000, dual="auto")
                    clf.fit(mu_i[:, np.newaxis], y_j)
                    preds = clf.predict(mu_i_test[:, np.newaxis])
                    score_matrix[i, j] = np.mean(preds == y_j_test)
                except:
                    score_matrix[i, j] = 0.  # fallback in edge cases

    return score_matrix


def compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
