import numpy as np
from utils import dataset_from_matrix
from .utils import _accuracy
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

class RejectOptionsLogisticLearner(object):
    def __init__(self, privileged_groups, unprivileged_groups, metric_name="Statistical parity difference", abs_bound=0.05):
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups
        self.metric_name = metric_name
        self.abs_bound = abs_bound
        self.threshold = 0.
        self.max_increase = 0.
        self.max_decrease = 0.
        self.no_increased = 0

    def debug_info(self):
        return str(self.threshold) + "\t" + str(self.max_increase) + "\t" + str(self.max_decrease) + "\t" + str(self.no_increased)

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())

        dataset_p = dataset.copy(deepcopy=True)
        dataset_p.scores = np.array(list(map(lambda x: [x[1]],reg.predict_proba(dataset.features))))
        #print(reg.predict_proba(dataset.features))
        #print(dataset_p.scores)
        dataset_p.labels = np.array(list(map(lambda x: [x], reg.predict(dataset.features))))

        ro = RejectOptionClassification(unprivileged_groups=self.unprivileged_group,
                privileged_groups=self.privileged_group,
                metric_name=self.metric_name,
                metric_ub=self.abs_bound,
                metric_lb=-self.abs_bound,low_class_thresh=0.45, high_class_thresh=.55)
        ro.fit(dataset, dataset_p)

        self.threshold = ro.classification_threshold

        def h(x):
            # add dummy labels as we're going to predict them anyway...
            x_with_labels = np.hstack((x, list(map(lambda x: [x], reg.predict(x)))))
            scores = list(map(lambda x:[x[1]],reg.predict_proba(x)))
            dataset_ = dataset_from_matrix(x_with_labels, dataset)
            dataset_.scores = np.array(scores)
            labels_pre = dataset_.labels

            dataset_ = ro.predict(dataset_)
            return dataset_.labels.ravel()

        def h_pr(x,boost=True):
            thresh = ro.classification_threshold
            scores = np.array(list(map(lambda x:x[1], reg.predict_proba(x))))
            orig_pred = list(map(lambda x: x[1] > thresh, reg.predict_proba(x)))
            boosted_pred = h(x)

            changed = (boosted_pred - orig_pred)

            group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
            _, priv_val = list(self.privileged_group[0].items())[0]
            grp_ind = dataset.feature_names.index(group_ft)


            bool_increased = np.where(changed == 1)
            max_increase = 0.
            if np.array(bool_increased).any():
                #print(bool_increased[0])
                self.no_increased = len(bool_increased[0])
                wrongly_flipped = (x[bool_increased,grp_ind] != unpriv_val).sum()
                if wrongly_flipped > 0:
                    raise Warning("RO flipped " + str(wrongly_flipped) + " labels (0 to 1) for privileged group")
                max_increase = (thresh-scores[bool_increased]).max()
                self.max_increase = max_increase
                #print("Increase", max_increase)
                #scores[x[:,grp_ind]==unpriv_val] += max_increase + 0.00001
                scores[bool_increased[0]] += max_increase + 0.00001


            bool_decreased = np.where(changed == -1)
            max_decrease = 0.
            if np.array(bool_decreased).any():
                #print(ro.classification_threshold - ro.ROC_margin)
                wrongly_flipped = (x[bool_decreased,grp_ind] != priv_val).sum()
                if wrongly_flipped > 0:
                    raise Warning("RO flipped " + str(wrongly_flipped) + " labels (1 to 0) for unprivileged group out of" + str(len(list(bool_decreased[0]))))
                max_decrease = (scores[bool_decreased] - thresh).max()
                self.max_decrease = max_decrease
                #scores[x[:,grp_ind]==priv_val] -= max_decrease + 0.000001
                scores[bool_decreased[0]] -= max_decrease + 0.00001
                #print("Decrease", max_decrease)

            boosted_pred = np.array(np.where(boosted_pred)[0])
            score_pred = np.array(np.where(scores>=thresh)[0])
            diff = np.setdiff1d(boosted_pred, score_pred)
            assert(len(diff)==0)



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
