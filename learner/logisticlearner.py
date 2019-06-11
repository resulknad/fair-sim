from sklearn.linear_model import LogisticRegression
from .utils import _drop_protected
import numpy as np
class LogisticLearner(object):
    threshold = 0.5
    def __init__(self, exclude_protected=False):
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        self.dataset = dataset
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

        self.coefs = (sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: -abs(x[1])))

        self.h = reg
        return None

    def predict(self, x):
        return self.h.predict(self.drop_prot(self.dataset, x))

    def predict_proba(self, x):
        return list(map(lambda x: x[1],self.h.predict_proba(self.drop_prot(self.dataset, x))))

    def drop_prot(self, dataset, x):
        return _drop_protected(dataset, np.array(x)) if self.exclude_protected else x

    def accuracy(self, dataset):
        return self.h.score(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

