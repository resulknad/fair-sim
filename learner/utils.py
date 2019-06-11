import pandas as pd
import numpy as np

# calc accuracy
def _accuracy(h, dataset):
    float_to_bool = lambda arr: np.array(list(map(lambda x: True if x == 1.0 else False, arr)))
    n_correct = (float_to_bool(h(dataset.features)) & float_to_bool(dataset.labels.ravel())).sum()
    return n_correct / len(dataset.labels.ravel())

# drop protected attributes from numpy array
def _drop_protected(dataset, features):
    ft_names = dataset.protected_attribute_names
    ft_indices = list(map(lambda x: not x in ft_names, dataset.feature_names))
    return np.array(features)[:,ft_indices]
