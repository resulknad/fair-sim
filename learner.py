from scipy.optimize import minimize
import numpy as np
from scipy import integrate
import pandas as pd
#import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from fairlearn.classred import expgrad
from fairlearn.moments import DP
from aif360.datasets import BinaryLabelDataset
from scipy.optimize import minimize
from simulation import Simulation

from aif360.metrics import BinaryLabelDatasetMetric
from scipy import optimize

from plot import _df_selection
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification

# util to calc accuracy
def _accuracy(h, dataset):
    float_to_bool = lambda arr: np.array(list(map(lambda x: True if x == 1.0 else False, arr)))
    n_correct = (float_to_bool(h(dataset.features, False)) & float_to_bool(dataset.labels.ravel())).sum()
    return n_correct / len(dataset.labels.ravel())

class RejectOptionsLogisticLearner(object):
    def __init__(self, privileged_groups, unprivileged_groups, metric_name="Statistical parity difference"):
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups
        self.metric_name = metric_name

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())

        dataset_p = dataset.copy(deepcopy=True)
        dataset_p.scores = np.array(list(map(lambda x: [x[1]],reg.predict_proba(dataset.features))))
        #print(reg.predict_proba(dataset.features))
        #print(dataset_p.scores)
        dataset_p.labels = np.array(list(map(lambda x: [x], reg.predict(dataset.features))))

        ro = RejectOptionClassification(unprivileged_groups=self.unprivileged_group, privileged_groups=self.privileged_group,metric_name=self.metric_name)
        ro.fit(dataset, dataset_p)

        def h(x, single=True):
            # add dummy labels as we're going to predict them anyway...
            x_with_labels = np.hstack((x, list(map(lambda x: [x], reg.predict(x)))))
            scores = list(map(lambda x:[x[1]],reg.predict_proba(x)))
            dataset_ = Simulation.dataset_from_matrix(x_with_labels, dataset)
            dataset_.scores = np.array(scores)
            labels_pre = dataset_.labels

            dataset_ = ro.predict(dataset_)
            return dataset_.labels.ravel()

        self.h = h
        return h


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
        #print(sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))
        #exit(1)

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

        self.h = lambda x,single=True: bc_binary(x)[0] if single else bc_binary(x)
        return self.h

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

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


class LogisticLearner(object):
    def __init__(self, exclude_protected=False):
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

        #print(sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))
        #exit(1)
        self.h = reg

        return lambda x,single=True: reg.predict(self.drop_prot(dataset, x))[0] if single else reg.predict(self.drop_prot(dataset, x))

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

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())



#class AdversialDebiasingLogisticLearner(object):
#    def __init__(self, **args):
#        self.args = args
#
#    def fit(self, dataset):
#        dataset.features = np.array(list(map(lambda x: np.array(x), dataset.features)))
#        self.args['sess'] = tf.Session()
#        self.args['scope_name'] = str(np.random.randint(1000000))
#
#        self.ad = AdversarialDebiasing(**self.args)
#
#        dataset_ = self.ad.fit_predict(dataset)
#
#
#        self.args['sess'].close()
#        tf.reset_default_graph()
#
#
#        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel())
#        self.h = reg
#
#        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)
#
#    def accuracy(self, dataset):
#        return self.h.score(dataset.features, dataset.labels.ravel())
#
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

import matplotlib.pyplot as plt
from itertools import product

class CoateLouryLearner(object):
    def __init__(self, f_u, f_q, x_u, x_q, pi_b, pi_w):
        self.f_u, self.f_q, self.x_u, self.x_q, self.pi_b, self.pi_w = f_u, f_q, x_u, x_q, pi_b, pi_w

    def fit(self, dataset):
        # do multivariate root finding:
        # 1. for b
        # 2. for w
        # 3. equal posterior probs.

        def bayes_optimal_classifier(x):
            # x = [theta_w, theta_b]

            # returns optimization goal for opt classifier
            def opt_fn_opt_classifier(pi):
                return lambda theta: (self.f_u(theta) *((1.-pi)*self.x_u)/(pi*self.x_q)) - (self.f_q(theta) )

            # returns optimization constraint for aff action
            def affirmative_action(pi):

                def fn(theta):
                    return pi*(1.-integrate.quad(self.f_q,0,theta)[0])+(1.-pi)*(1-integrate.quad(self.f_u,0,theta)[0])

                return fn
            opt_classifier_w = opt_fn_opt_classifier(self.pi_w)
            opt_classifier_b = opt_fn_opt_classifier(self.pi_b)
            opt_aff_action = lambda theta_w, theta_b: affirmative_action(self.pi_w)(theta_w) - affirmative_action(self.pi_b)(theta_b)




            return ([(opt_classifier_w(x[0])), (opt_classifier_b(x[1])), 0])#(opt_aff_action(x[0],x[1]))])



        def bruce_force_minimization(fn, n, no):
            min_val, min_x = [1000]*no, []
            for x in product(*([list(np.linspace(0,1,500))]*n)):
                if [abs(a) for a in min_val] >= [abs(a) for a in fn(x)] and fn(x)[0] <=0 and fn(x)[1]<=0 and abs(fn(x)[2]) < 0.5:
                    min_val, min_x = fn(x), x
            print(min_val)
            return min_x

        sol = bruce_force_minimization(bayes_optimal_classifier, 2, 3)
        print(sol)



        #print(cross_val_score(MultinomialNB(), X, y, cv=5))
        #exit()
        #reg = MultinomialNB().fit(X,y)
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())
        self.h = reg

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())

def _drop_protected(dataset, features):
    ft_names = dataset.protected_attribute_names
    ft_indices = list(map(lambda x: not x in ft_names, dataset.feature_names))
    return np.array(features)[:,ft_indices]
