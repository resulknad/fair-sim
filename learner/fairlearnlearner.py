from sklearn.linear_model import LogisticRegression
from fairlearn.classred import expgrad
from fairlearn.moments import DP
from .utils import _accuracy
import pandas as pd
import numpy as np

class FairLearnLearner(object):
    threshold = 0.5
    def __init__(self, privileged_group, unprivileged_group):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def fit(self, dataset):
        # sanity checks
        assert(len(self.privileged_group)==1)
        assert((self.privileged_group[0].keys()==self.unprivileged_group[0].keys()))
        class_attr = list(self.privileged_group[0].keys())[0]

        reg = LogisticRegression(solver='liblinear',max_iter=1000000000)

        class_ind = dataset.feature_names.index(class_attr)
        X = pd.DataFrame(dataset.features)
        A = pd.Series(dataset.features[:,class_ind])
        Y = pd.Series(dataset.labels.ravel())

        bc = expgrad(X, A, Y, reg, nu=1, cons=DP()).best_classifier
        bc_binary = lambda x: (list(bc(x) > 0.5))

        self.bc = bc
        self.bc_binary = bc_binary

    def predict_proba(self, x):
        return np.array(self.bc(x))

    def predict(self, x):
        return self.bc_binary(x)

    def accuracy(self, dataset):
        return _accuracy(self.bc_binary, dataset)
