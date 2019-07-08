import numpy as np
from utils import dataset_from_matrix
from .utils import _accuracy, _drop_protected
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

from .generallearner import GeneralLearner

class RejectOptionsLogisticLearner(GeneralLearner):
    """Reject Options post-processing technique is applied on a logistic regression model."""
    def __init__(self, privileged_groups, unprivileged_groups, metric_name="Statistical parity difference", abs_bound=0.05, exclude_protected=False):
        self.exclude_protected = exclude_protected
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups
        self.metric_name = metric_name
        self.abs_bound = abs_bound
        self.threshold = 0.
        self.max_increase = 0.
        self.max_decrease = 0.
        self.no_increased = 0


    def drop_prot(self, dataset, x):
        return _drop_protected(dataset, np.array(x)) if self.exclude_protected else x

    def debug_info(self):
        return str(self.threshold) + "\t" + str(self.max_increase) + "\t" + str(self.max_decrease) + "\t" + str(self.no_increased)

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000).fit(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

        dataset_p = dataset.copy(deepcopy=True)
        dataset_p.scores = np.array(list(map(lambda x: [x[1]],reg.predict_proba(self.drop_prot(dataset, dataset.features)))))
        #print(reg.predict_proba(dataset.features))
        #print(dataset_p.scores)
        dataset_p.labels = np.array(list(map(lambda x: [x], reg.predict(self.drop_prot(dataset, dataset.features)))))

        ro = RejectOptionClassification(unprivileged_groups=self.unprivileged_group,
                privileged_groups=self.privileged_group,
                metric_name=self.metric_name,
                metric_ub=self.abs_bound,
                metric_lb=-self.abs_bound,low_class_thresh=0.45, high_class_thresh=.55)
        ro.fit(dataset, dataset_p)

        self.threshold = ro.classification_threshold

        def h(x):
            # add dummy labels as we're going to predict them anyway...
            x_with_labels = np.hstack((x, list(map(lambda x: [x], reg.predict(self.drop_prot(dataset, x))))))
            scores = list(map(lambda x:[x[1]],reg.predict_proba(self.drop_prot(dataset, x))))
            dataset_ = dataset_from_matrix(x_with_labels, dataset)
            dataset_.scores = np.array(scores)
            labels_pre = dataset_.labels

            dataset_ = ro.predict(dataset_)
            return dataset_.labels.ravel()

        def h_pr(x,boost=True):
            thresh = ro.classification_threshold
            scores = np.array(list(map(lambda x:x[1], reg.predict_proba(self.drop_prot(dataset, x)))))
            scores_ = np.array(list(map(lambda x:x[1], reg.predict_proba(self.drop_prot(dataset, x)))))
            orig_pred = list(map(lambda x: x[1] > thresh, reg.predict_proba(self.drop_prot(dataset, x))))
            boosted_pred = h(x)

            changed = (boosted_pred - orig_pred)

            group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
            _, priv_val = list(self.privileged_group[0].items())[0]
            grp_ind = dataset.feature_names.index(group_ft)

            priv_ind = x[:,grp_ind] != unpriv_val
            unpriv_ind = x[:,grp_ind] == unpriv_val

            lower_bound = ro.classification_threshold - ro.ROC_margin -0.1
            upper_bound = ro.classification_threshold + ro.ROC_margin + 0.1
            #print(self.metric_name, lower_bound, upper_bound)

            def booster_fn(scores):
                return (expit(75*(scores - lower_bound)) - expit(75*(scores - upper_bound))) * ro.ROC_margin

            scores[priv_ind] -= booster_fn(scores[priv_ind])

            scores[unpriv_ind] += booster_fn(scores[unpriv_ind])

            assert((np.clip(scores, None, 1.)[priv_ind] <= scores_[priv_ind]).all())
            assert((scores != scores_).any())


            boosted_pred = np.array(np.where(boosted_pred)[0])
            score_pred = np.array(np.where(scores>=thresh)[0])
            diff = np.setdiff1d(boosted_pred, score_pred)

            #print(ro.classification_threshold, ro.ROC_margin)
            #print(diff, scores[diff], scores_[diff])
            #print(booster_fn(scores_[diff]))
            #print(list(map(lambda x: x-thresh, scores[diff])))
            #assert(len(diff)==0)



            return np.clip(scores, None, 1.) if boost else np.array(list(map(lambda x:x[1], reg.predict_proba(x))))

        self.h_pr = h_pr
        self.h = h

        def bak():
            bool_increased = np.where(changed == 1)
            bool_decreased = np.where(changed == -1)
            if self.threshold == 0:
                max_increase = (thresh-scores[bool_increased]).max()
                max_decrease = (scores[bool_decreased] - thresh).max()

                self.threshold = thresh
                self.max_increase = max_increase
                self.max_decrease = max_decrease


            scores[x[:,grp_ind]==unpriv_val] += self.max_increase + 0.00001
            scores[x[:,grp_ind]==priv_val] -= self.max_decrease + 0.000001


            #boosted_pred = np.array(np.where(boosted_pred)[0])
            #score_pred = np.array(np.where(scores>=thresh)[0])
            #diff = np.setdiff1d(boosted_pred, score_pred)
            #assert(len(diff)==0)


            return np.clip(scores, None, 1.)


    def predict_proba(self, x, boost=True):
        return self.h_pr(x, boost)

    def predict(self, x):
        return self.h(x)

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)
