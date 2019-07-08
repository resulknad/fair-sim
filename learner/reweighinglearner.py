from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression

from .generallearner import GeneralLearner

class ReweighingLogisticLearner(GeneralLearner):
    """Instances are weighted s.t. statistical parity difference on training set is 0. Afterwards a logistic regression model is fitted on the weighted dataset."""
    threshold = 0.5
    def __init__(self, privileged_group, unprivileged_group):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def fit(self, dataset):
        RW = Reweighing(unprivileged_groups=self.unprivileged_group,
                privileged_groups=self.privileged_group)

        mean_diff_metric = lambda dataset: BinaryLabelDatasetMetric(dataset,
                                             unprivileged_groups=self.unprivileged_group,
                                             privileged_groups=self.privileged_group).mean_difference()
        dataset_ = RW.fit_transform(dataset)

        print("before reweighing (meandiff):",mean_diff_metric(dataset),"after:",mean_diff_metric(dataset_))

        #reg_ = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel())

        reg = LogisticRegression(solver='liblinear',max_iter=1000000000).fit(dataset_.features, dataset_.labels.ravel(), sample_weight=dataset_.instance_weights)
        #print("reweighted",sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))

        #print(sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))

        self.h = reg

    def predict_proba(self, x):
        return list(map(lambda x: x[1],self.h.predict_proba(x)))

    def predict(self, x):
        return self.h.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())

