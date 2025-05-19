"""Implementation of Disentanglement, Completeness and Informativeness (DCI)."""
import numpy as np
import scipy
from sklearn import ensemble
import logging
import gin
import metrics.utils as utils


@gin.configurable(
    "dci",
    denylist=["ground_truth_data", "representation_function", "random_state", "artifact_dir"])
def compute_dci(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=64):
    """Computes the DCI scores."""
    del artifact_dir
    logging.info("Generating training and test sets.")

    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train, random_state, batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test, random_state, batch_size)

    return _compute_dci(mus_train, ys_train, mus_test, ys_test)


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    scores = {}
    importance_matrix, train_acc, test_acc = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)

    scores["informativeness_train"] = train_acc
    scores["informativeness_test"] = test_acc
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros((num_codes, num_factors), dtype=np.float64)
    train_acc = []
    test_acc = []

    for i in range(num_factors):
        y = y_train[i]
        if len(np.unique(y)) < 2:
            # Skip constant factor (like color)
            continue

        clf = ensemble.GradientBoostingClassifier()
        try:
            clf.fit(x_train.T, y)
        except ValueError as e:
            print(f"Skipping factor {i}: {e}")
            continue

        importance_matrix[:, i] = np.abs(clf.feature_importances_)
        train_acc.append(np.mean(clf.predict(x_train.T) == y_train[i]))
        test_acc.append(np.mean(clf.predict(x_test.T) == y_test[i]))

    return importance_matrix, np.mean(train_acc), np.mean(test_acc)



def disentanglement(importance_matrix):
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0:
        return 0.0
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def disentanglement_per_code(importance_matrix):
    return 1.0 - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def completeness(importance_matrix):
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0:
        return 0.0
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def completeness_per_factor(importance_matrix):
    return 1.0 - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])
