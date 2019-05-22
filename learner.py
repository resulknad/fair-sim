from scipy.optimize import minimize
import numpy as np
from scipy import integrate
import pandas as pd
#import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from fairlearn.classred import expgrad
from fairlearn.moments import DP
from aif360.datasets import BinaryLabelDataset
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product
from aif360.metrics import BinaryLabelDatasetMetric
from scipy import optimize
from plot import _df_selection
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from simulation import Simulation

# util to calc accuracy
def _accuracy(h, dataset):
    float_to_bool = lambda arr: np.array(list(map(lambda x: True if x == 1.0 else False, arr)))
    n_correct = (float_to_bool(h(dataset.features)) & float_to_bool(dataset.labels.ravel())).sum()
    return n_correct / len(dataset.labels.ravel())

class RejectOptionsLogisticLearner(object):
    def __init__(self, privileged_groups, unprivileged_groups, metric_name="Statistical parity difference", abs_bound=0.05):
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups
        self.metric_name = metric_name
        self.abs_bound = abs_bound

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
                metric_lb=-self.abs_bound)
        ro.fit(dataset, dataset_p)

        def h(x):
            # add dummy labels as we're going to predict them anyway...
            x_with_labels = np.hstack((x, list(map(lambda x: [x], reg.predict(x)))))
            scores = list(map(lambda x:[x[1]],reg.predict_proba(x)))
            dataset_ = Simulation.dataset_from_matrix(x_with_labels, dataset)
            dataset_.scores = np.array(scores)
            labels_pre = dataset_.labels

            dataset_ = ro.predict(dataset_)
            return dataset_.labels.ravel()

        def h_pr(x):
            thresh = ro.classification_threshold
            scores = np.array(list(map(lambda x:x[1], reg.predict_proba(x))))
            orig_pred = list(map(lambda x: x[1] > thresh, reg.predict_proba(x)))
            boosted_pred = h(x)

            changed = (boosted_pred - orig_pred)

            group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
            _, priv_val = list(self.privileged_group[0].items())[0]
            grp_ind = dataset.feature_names.index(group_ft)


            bool_increased = np.where(changed == 1)
            if np.array(bool_increased).any():
                wrongly_flipped = (x[bool_increased,grp_ind] != unpriv_val).sum()
                if wrongly_flipped > 0:
                    raise Warning("RO flipped " + str(wrongly_flipped) + " labels (0 to 1) for privileged group")
                max_increase = (thresh-scores[bool_increased]).max()
                scores[x[:,grp_ind]==unpriv_val] += max_increase + 0.00001
                #print("Increase", max_increase)


            bool_decreased = np.where(changed == -1)
            if np.array(bool_decreased).any():
                wrongly_flipped = (x[bool_decreased,grp_ind] != priv_val).sum()
                if wrongly_flipped > 0:
                    raise Warning("RO flipped " + str(wrongly_flipped) + " labels (1 to 0) for unprivileged group out of" + str(len(list(bool_decreased[0]))))
                max_decrease = (scores[bool_decreased] - thresh).max()
                scores[x[:,grp_ind]==priv_val] -= max_decrease + 0.000001
                #print("Decrease", max_decrease)

            boosted_pred = np.array(np.where(boosted_pred)[0])
            score_pred = np.array(np.where(scores>=thresh)[0])
            diff = np.setdiff1d(boosted_pred, score_pred)
            assert(len(diff)==0)

            return np.clip(scores, None, 1.)


        self.h_pr = h_pr
        self.h = h

    def predict_proba(self, x):
        return self.h_pr(x)

    def predict(self, x):
        return self.h(x)

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

class StatisticalParityFlipperLogisticLearner(object):
    def __init__(self, privileged_group, unprivileged_group, exclude_protected=False):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.ratio = 0.5
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        def drop_prot(x):
            return _drop_protected(dataset, x) if self.exclude_protected else x

        reg = LogisticRegression(solver='lbfgs',max_iter=1000000000, C=1000000000000000000000.0).fit(drop_prot(dataset.features), dataset.labels.ravel())
        self.h = reg

        assert(len(self.privileged_group)==1)
        assert(len(dataset.label_names) == 1)

        #df = dataset.convert_to_dataframe()[0].drop(columns=dataset.label_names)

        dataset.features = np.array(list(map(np.array, dataset.features)))
        df = pd.DataFrame(data=dataset.features, columns=dataset.feature_names)
        # priv count
        #print(self.privileged_group[0])
        n_p = len(_df_selection(df, self.privileged_group[0]).values)

        # not priv count
        n_np = len(_df_selection(df, self.unprivileged_group[0]).values)

        data = _df_selection(df, self.privileged_group[0])
        probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
        n_p_y = (np.array(probs)>0.5).sum()

        data = _df_selection(df, self.unprivileged_group[0])
        assert(len(data.values)<=n_np)
        probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
        n_np_y = (np.array(probs)>0.5).sum()

        stat_par_diff = n_p_y/n_p - n_np_y/n_np

        # unprivileged should be unprivileged...

        group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
        grp_i = dataset.feature_names.index(group_ft)

        def decision_fn(x,single=True):
            x = np.array(x)
            y_pred = reg.predict(drop_prot(x))


            def flip(grp_val, flip_from, flip_to, fraction):
                # select all from group in x
                grp_ind = x[:,grp_i] == grp_val
                # find indices we predicted flip_from
                label_ind = (y_pred == flip_from)
                # flippable indices
                flippable = (grp_ind & label_ind).nonzero()[0]
                # shuffle
                flippable = np.random.permutation(flippable)
                # truncate
                truncated_size = int(round(abs(stat_par_diff)*grp_ind.sum()*fraction))
                #print(grp_val,"stat par diff is", stat_par_diff, " so we flip ",truncated_size)
                flippable = flippable[:truncated_size]

                # flip
                y_pred[flippable] = [flip_to]* len(flippable)

            if stat_par_diff > 0:
                flip(unpriv_val, 0, 1, 1)
                #flip(1-unpriv_val, 1, 0, .5)
            else:
                flip(unpriv_val, 1, 0, 1)
                #flip(1-unpriv_val, 0, 1, .5)

            #print("flipped ", len(flippable))


            return y_pred.tolist()

        #if stat_par_diff > 0:
            #print("flipping up from 0 to 1 with pr", stat_par_diff/2., " for p from 1 to 0 with same pr")
        #else:
            #print("flipping up from 1 to 0 with pr", stat_par_diff/2., " for p from 0 to 1 with same pr")


        self.h = decision_fn
        return decision_fn #lambda x: 1 if np.add(reg.predict_proba(x)[1],x[grp_i]*boost > 0.5 else 0

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

class StatisticalParityLogisticLearner(object):
    def __init__(self, privileged_group, unprivileged_group, eps, exclude_protected=False):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.eps = eps
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        def drop_prot(x):
            return _drop_protected(dataset, x) if self.exclude_protected else x
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(drop_prot(dataset.features), dataset.labels.ravel())
        self.h = reg

        assert(len(self.privileged_group)==1)
        assert(len(dataset.label_names) == 1)

        #df = dataset.convert_to_dataframe()[0].drop(columns=dataset.label_names)

        dataset.features = np.array(list(map(np.array, dataset.features)))
        df = pd.DataFrame(data=dataset.features, columns=dataset.feature_names)
        # priv count
        #print(self.privileged_group[0])
        n_p = len(_df_selection(df, self.privileged_group[0]).values)

        # not priv count
        n_np = len(_df_selection(df, self.unprivileged_group[0]).values)

        def stat_parity_diff(boost):
            data = _df_selection(df, self.privileged_group[0])
            probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
            n_p_y = (np.array(probs)>0.5).sum()

            data = _df_selection(df, self.unprivileged_group[0])
            assert(len(data.values)<=n_np)
            probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
            n_np_y = (np.array(np.add(probs,[boost]*len(probs)))>0.5).sum()


            stat_par_diff = n_np_y/n_np -  n_p_y/n_p
            if boost == 0:
                return stat_par_diff
            else:
                return stat_par_diff# - stat_parity_diff(0)/2.
            #print("Boost:",boost,n_np_y,"of", n_np, n_p_y,"of",n_p,stat_par_diff)
            return stat_par_diff
        try:
            boost = optimize.bisect(stat_parity_diff, 0, 1, xtol=self.eps, disp=True)
        except ValueError: #
            print("couldnt find appropriate boost, dont boost")
            boost = 0
        #boost = 0.
        print("Boost:",boost)

        group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
        grp_i = dataset.feature_names.index(group_ft)
        def decision_fn(x,single=True):
            ys = []
            probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(x))))
            for x,p in zip(x,probs):
                p_ = p + boost if x[grp_i] == unpriv_val else p
                if p_ > 0.5:
                    ys.append(1)
                else:
                    ys.append(0)
            return ys[0] if single else ys

        self.h = decision_fn
        return decision_fn #lambda x: 1 if np.add(reg.predict_proba(x)[1],x[grp_i]*boost > 0.5 else 0

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

class FairLearnLearner(object):
    def __init__(self, privileged_group, unprivileged_group):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def fit(self, dataset):
        # sanity checks
        assert(len(self.privileged_group)==1)
        assert((self.privileged_group[0].keys()==self.unprivileged_group[0].keys()))
        class_attr = list(self.privileged_group[0].keys())[0]

        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0)

        class_ind = dataset.feature_names.index(class_attr)
        X = pd.DataFrame(dataset.features)
        A = pd.Series(dataset.features[:,class_ind])
        Y = pd.Series(dataset.labels.ravel())

        bc = expgrad(X, A, Y, reg, nu=2.9e-12, cons=DP()).best_classifier
        bc_binary = lambda x: (list(bc(x) > 0.5))

        self.bc = bc
        self.bc_binary = bc_binary

    def predict_proba(self, x):
        return self.bc(x)

    def predict(self, x):
        return self.bc_binary(x)

    def accuracy(self, dataset):
        return _accuracy(self.bc_binary, dataset)

class PrejudiceRemoverLearner(object):
    def __init__(self, privileged_group, unprivileged_group):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def fit(self, dataset):
        # sanity checks
        assert(len(self.privileged_group)==1)
        assert((self.privileged_group[0].keys()==self.unprivileged_group[0].keys()))

        class_attr = list(self.privileged_group[0].keys())[0]

        # TODO: include sensitive attr?
        pr = PrejudiceRemover(eta=1.0, sensitive_attr='group', class_attr='y')
        pr.fit(dataset)
        def h(x, single=True):
            # add dummy labels, we'll predict them now anyway
            x_with_labels = np.hstack((x, [[0]] * len(x)))
            # construct dataset
            dataset_ = Simulation.dataset_from_matrix(x_with_labels, dataset)

            dataset_p = pr.predict(dataset_)
            return dataset_p.labels.ravel()
        self.h = h
        return h

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

class RandomForestLearner(object):
    def __init__(self, exclude_protected=False):
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        self.dataset = dataset
        reg = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=1).fit(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

        self.h = reg

    def predict(self, x):
        return self.h.predict(self.drop_prot(self.dataset, x))

    def predict_proba(self, x):
        return list(map(lambda x: x[1],self.h.predict_proba(self.drop_prot(self.dataset, x))))

    def drop_prot(self, dataset, x):
        return _drop_protected(dataset, np.array(x)) if self.exclude_protected else x

    def accuracy(self, dataset):
        return self.h.score(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

class GaussianNBLearner(object):
    def __init__(self, exclude_protected=False):
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        self.dataset = dataset
        reg = GaussianNB().fit(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

        #print(sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))
        #exit(1)
        self.h = reg

    def predict(self, x):
        return self.h.predict(self.drop_prot(self.dataset, x))

    def predict_proba(self, x):
        return list(map(lambda x: x[1],self.h.predict_proba(self.drop_prot(self.dataset, x))))

    def drop_prot(self, dataset, x):
        return _drop_protected(dataset, np.array(x)) if self.exclude_protected else x

    def accuracy(self, dataset):
        return self.h.score(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

class LogisticLearner(object):
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


class ReweighingLogisticLearner(object):
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

        #print("before reweighing (meandiff):",mean_diff_metric(dataset),"after:",mean_diff_metric(dataset_))


        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel(), sample_weight=dataset_.instance_weights)
        #print("reweighted",sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel())
        #print(sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))

        self.h = reg

    def predict_proba(self, x):
        return list(map(lambda x: x[1],self.h.predict_proba(x)))

    def predict(self, x):
        return self.h.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())

class EqOddsPostprocessingLogisticLearner(object):
    def __init__(self, privileged_groups, unprivileged_groups):
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())

        dataset_p = dataset.copy()
        dataset_p.scores = np.array(list(map(lambda x: x[1],reg.predict_proba(dataset.features))))
        #dataset_p.labels = np.array(list(map(lambda x: [x], reg.predict(dataset.features))))

        eqodds = EqOddsPostprocessing(unprivileged_groups=self.unprivileged_group, privileged_groups=self.privileged_group)
        eqodds.fit(dataset, dataset_p)

        def h(x, single=True):
            # library does not support datasets with single instances
            if single:
                raise NotImplementedError

            # add dummy labels as we're going to predict them anyway...
            x_with_labels = np.hstack((x, list(map(lambda x: [x], reg.predict(x)))))
            scores = list(map(lambda x:x[1],reg.predict_proba(x)))
            if single:
                assert(len(x_with_labels) == 1)
                x_with_labels = np.repeat(x_with_labels, 100, axis=0)
                scores = np.repeat(scores, 100, axis=0)
            dataset_ = Simulation.dataset_from_matrix(x_with_labels, dataset)
            dataset_.scores = np.array(scores)
            labels_pre = dataset_.labels

            dataset_ = eqodds.predict(dataset_)

            #if not (labels_pre == dataset_.labels).all():
            #    print("fav:",dataset_.favorable_label)
            #    print("labels did change after eqodds.", labels_pre[0][0],"to", dataset_.labels,"for group",x_with_labels[0][1])
            #else:
            #    print("did not change", len(x_with_labels))
            return dataset_.labels.ravel()

        self.h = h
        return h


    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)



def _drop_protected(dataset, features):
    ft_names = dataset.protected_attribute_names
    ft_indices = list(map(lambda x: not x in ft_names, dataset.feature_names))
    return np.array(features)[:,ft_indices]
