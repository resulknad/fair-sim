import pandas as pd
import numpy as np

# calc accuracy
def _accuracy(h, dataset):
    """Utility function to calculate accuracy.

    :param h: Classification function
    :param dataset: Dataset containing features and true labels
    :returns: Accuracy"""
    float_to_bool = lambda arr: np.array(list(map(lambda x: True if x == 1.0 else False, arr)))
    n_correct = (float_to_bool(h(dataset.features)) & float_to_bool(dataset.labels.ravel())).sum()
    return n_correct / len(dataset.labels.ravel())

# drop protected attributes from numpy array
def _drop_protected(dataset, features):
    """Utility function to drop protected attributes from feature matrix `features`.

    :param dataset: AIF360 dataset containing information about protected features..
    :param features: Feature matrix (dimension: `n_instances x n_features`)
    :returns: Modified feature matrix (dimension: `n_instances x (n_features - n_protected_features)`)"""
    ft_names = dataset.protected_attribute_names
    ft_indices = list(map(lambda x: not x in ft_names, dataset.feature_names))
    return np.array(features)[:,ft_indices]
