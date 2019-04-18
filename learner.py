from scipy.optimize import minimize
import numpy as np
from scipy import integrate
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset
from scipy.optimize import minimize
from simulation import Simulation

from aif360.metrics import BinaryLabelDatasetMetric
from scipy import optimize

from plot import _df_selection

# util to calc accuracy
def _accuracy(h, dataset):
    float_to_bool = lambda arr: np.array(list(map(lambda x: True if x == 1.0 else False, arr)))
    n_correct = (float_to_bool(h(dataset.features, False)) & float_to_bool(dataset.labels.ravel())).sum()
    return n_correct / len(dataset.labels.ravel())

class StatisticalParityLogisticLearner(object):
    def __init__(self, privileged_group, unprivileged_group, eps):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.eps = eps

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())
        self.h = reg

        assert(len(self.privileged_group)==1)
        assert(len(dataset.label_names) == 1)

        #df = dataset.convert_to_dataframe()[0].drop(columns=dataset.label_names)

        dataset.features = np.array(list(map(np.array, dataset.features)))
        df = pd.DataFrame(data=dataset.features, columns=dataset.feature_names)
        # priv count
        print(self.privileged_group[0])
        n_p = len(_df_selection(df, self.privileged_group[0]).values)

        # not priv count
        n_np = len(_df_selection(df, self.unprivileged_group[0]).values)

        def stat_parity_diff(boost):
            data = _df_selection(df, self.privileged_group[0])
            probs = list(map(lambda x: x[1], reg.predict_proba(data.values)))
            n_p_y = (np.array(probs)>0.5).sum()

            data = _df_selection(df, self.unprivileged_group[0])
            assert(len(data.values)<=n_np)
            probs = list(map(lambda x: x[1], reg.predict_proba(data.values)))
            n_np_y = (np.array(np.add(probs,[boost]*len(probs)))>0.5).sum()


            stat_par_diff = n_np_y/n_np - n_p_y/n_p
            #print("Boost:",boost,n_np_y,"of", n_np, n_p_y,"of",n_p,stat_par_diff)
            return stat_par_diff

        boost = optimize.bisect(stat_parity_diff, 0, 1, xtol=self.eps, disp=True)
        #print("Boost:",boost)

        group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
        grp_i = dataset.feature_names.index(group_ft)
        def decision_fn(x,single=True):
            ys = []
            probs = list(map(lambda x: x[1], reg.predict_proba(x)))
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



class LogisticLearner(object):
    def fit(self, dataset):
        #print(cross_val_score(MultinomialNB(), X, y, cv=5))
        #exit()
        #reg = MultinomialNB().fit(X,y)
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())
        self.h = reg

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())


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

        print("before reweighing (meandiff):",mean_diff_metric(dataset),"after:",mean_diff_metric(dataset_))


        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel(), sample_weight=dataset_.instance_weights)
        self.h = reg

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())

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

        print("before reweighing (meandiff):",mean_diff_metric(dataset),"after:",mean_diff_metric(dataset_))


        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel(), sample_weight=dataset_.instance_weights)
        self.h = reg

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

    def accuracy(self, dataset):
        return self.h.score(dataset.features, dataset.labels.ravel())


class AdversialDebiasingLogisticLearner(object):
    def __init__(self, **args):
        self.args = args

    def fit(self, dataset):
        dataset.features = np.array(list(map(lambda x: np.array(x), dataset.features)))
        self.args['sess'] = tf.Session()
        self.args['scope_name'] = str(np.random.randint(1000000))

        self.ad = AdversarialDebiasing(**self.args)

        dataset_ = self.ad.fit_predict(dataset)


        self.args['sess'].close()
        tf.reset_default_graph()


        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset_.features, dataset_.labels.ravel())
        self.h = reg

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

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

        eqodds = CalibratedEqOddsPostprocessing(unprivileged_groups=self.unprivileged_group, privileged_groups=self.privileged_group, cost_constraint='fpr')
        eqodds.fit(dataset, dataset_p)

        def h(x, single=True):
#            df,_ = dataset.convert_to_dataframe()
#            df.update(x)
#            dataset_ = BinaryLabelDataset(df=df, label_names=dataset.label_names, protected_attribute_names=dataset.protected_attribute_names)

#            dataset_ = dataset.align_datasets(dataset_)
#            dataset_.validate_dataset()


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
            # library does not support datasets with single instances
            # workaround to extract probability w/o accessing stuff from the implementation
            if single:
                unique, counts = np.unique(dataset_.labels.ravel(), return_counts=True)
                val, occurences = list(dict(zip(unique, counts)).items())[0]
                random_int = np.random.randint(1, 101)
                if random_int <= occurences:
                    print(random_int,occurences,"deciding",val)
                    dataset_.labels = [[val]]
                else:
                    dataset_.labels = [[1-val]]
                    print(random_int,occurences,"deciding",1-val)
                    # assuming (!) binary labels

            if not (labels_pre == dataset_.labels).all() and single:
                print("fav:",dataset_.favorable_label)
                print("labels did change after eqodds.", labels_pre[0][0],"to", dataset_.labels,"for group",x_with_labels[0][1])
            else:
                print("did not change", len(x_with_labels))
            return dataset_.labels[0] if single else dataset_.labels.ravel()
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
