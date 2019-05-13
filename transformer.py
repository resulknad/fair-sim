import matplotlib.pyplot as plt
from itertools import starmap
from functools import reduce
import time
import traceback

from aif360.algorithms import Transformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import uuid

# copied from SCIKIT OPTIMIZE
def approx_fprime(xk, f, epsilon, args=(), f0=None, immutable=[]):
    if f0 is None:
        f0 = f(*((xk,) + args))
    n_instances, n_features = xk.shape
    grad = np.zeros(xk.shape, float)
    ei = np.zeros(xk.shape, float)
    for k in range(n_features):
        if k not in immutable:
            ei[:,k] = 1.0
            d = epsilon * ei
            #print("d shape", d.shape, "xk shape", xk.shape)
            grad[:,k] = (f(*((xk + d,) + args)) - f0) / d[0][k]
            ei[k] = 0.0

    return grad

class AgentTransformer(Transformer):
# this class is not thread safe
    def __init__(self, agent_class, h, cost_distribution, scaler, no_neighbors=51, collect_incentive_data=False, avg_out_incentive=1, cost_distribution_dep=None, use_rank=True):

        self.avg_out_incentive = avg_out_incentive
        self.use_rank = True
        self.agent_class = agent_class
        self.h = h
        self.cost_distribution = cost_distribution
        self.no_neighbors = no_neighbors
        self.collect_incentive_data = collect_incentive_data
        self.cost_distribution_dep = cost_distribution_dep
        self.incentives = []

        super(AgentTransformer, self).__init__(
            agent_class=agent_class,
            h=h,
            cost_distribution=cost_distribution)

    def _optimal_x_gd(self, dataset, cost):
        X = dataset.features
        y = dataset.labels

        assert(len(dataset.protected_attribute_names)==1)
        immutable = [dataset.feature_names.index(dataset.protected_attribute_names[0])]

        # x0
        a = self.agent_class(self.h, dataset, cost, X, y)

        # max tracking
        #print(x)

        eps = 0.1
        gradient = approx_fprime(X, a.incentive, eps, immutable=immutable)
        print(gradient.shape)

        #print(gradient)
        #print("incentive", a.incentive(x))
        breakit = False

        # calculates bounds for clipping
        # if immutable, then lower bound = upper bound = present value
        min_domain = np.array(list(starmap(
                lambda i,ft: np.repeat(min(dataset.domains[ft]),len(X)) if ft in dataset.domains
                else X[:,i],
            zip(range(len(dataset.feature_names)),dataset.feature_names))))

        max_domain = np.array(list(starmap(
                lambda i,ft: np.repeat(max(dataset.domains[ft]),len(X)) if ft in dataset.domains
                else X[:,i],
            zip(range(len(dataset.feature_names)),dataset.feature_names))))

        min_domain = min_domain.transpose()
        max_domain = max_domain.transpose()

        incentive_last = 0
        incentive = a.incentive(X)
        for i in range(100):
            #print(X)
            incentive_last = incentive
            #print("Iteration ",i)
            X = np.add(X,gradient)
            X = np.clip(X, min_domain, max_domain)
            gradient = approx_fprime(X, a.incentive, eps, immutable=immutable)
            #print(np.mean(dict(zip(dataset.feature_names, gradient.transpose()))['investment_as_income_percentage']))
            #print(np.mean(dict(zip(dataset.feature_names, min_domain))['investment_as_income_percentage']))
            incentive = a.incentive(X)
            if ((incentive-incentive_last) < 0.001).all():
                print("Quit after ",i,"iterations")
                break


        X = dataset.enforce_dummy_coded(X)
        X_ = []
        for x, ft in zip(X.transpose(), dataset.feature_names):
            if ft in dataset.domains:
                d = sorted(dataset.domains[ft])
                x_mod = []
                for x_ in x:
                    loc = np.searchsorted(d, x_)
                    if loc > 0:
                        x__ = d[loc] if abs(d[loc]-x_) < abs(d[loc-1]-x_) else d[loc-1]
                        #print("picked ", x__, " because ", d[loc], " and ", d[loc-1], "orig",x_)
                    else:
                        x__ = d[loc]
                    x_mod.append(x__)
                X_.append(np.array(x_mod))
            else:
                X_.append(x)

        X_ = np.array(X_).transpose()


        return a.incentive(X_), X_


    def _do_simulation(self, dataset, gradient_descent=True):

        # setup incentive data collection
        if self.collect_incentive_data:
            self.incentives = []

        # fixed cost may be same for all instances
        if self.cost_distribution_dep is None:
            cost = self.cost_distribution(len(dataset.features))
        else:
        # or different depending on features (like group)
            cost = np.array(list(map(self.cost_distribution_dep, dataset.features)))

        dataset_ = dataset.copy(deepcopy=True)
        features_ = []
        labels_ = []
        changed_indices = []

        i=0
        grp0 = []
        grp1 = []

        incentives, X = self._optimal_x_gd(dataset, cost)

        for x_vec,x,y,c,incentive in zip(X,dataset.features, dataset.labels, cost, incentives):

            #print(incentive,x_vec)

            if incentive > 0 and not (x_vec == x).all():
                features_.append(np.array(x_vec))
                changed_indices.append(i)
                labels_.append([])

            else:
                features_.append(np.array(x))
                labels_.append(y)
            i+=1
            #print(features_)
        dataset_.features = features_

        #print("grp0: avg opt x",np.average(np.array(grp0)))
        #print("grp1: avg opt x",np.average(np.array(grp1)))
        X = np.array(features_)
        Y = np.array(labels_)

        # no changes during simulation
        # no need to assign new labels with KNN
        if len(changed_indices) == 0:
            dataset_.features = X
            dataset_.labels = np.array(Y.tolist())
            return dataset_

        unchanged_indices = np.setdiff1d(list(range(len(X))), changed_indices)
        X_changed = X[changed_indices,:]
        Y_changed = Y[changed_indices]

        X_unchanged = X[unchanged_indices,:]
        Y_unchanged = Y[unchanged_indices]

        assert(len(X_changed)==len(changed_indices))
        assert(len(X_unchanged)==len(X)-len(changed_indices))

        assert(len(dataset.protected_attribute_names)==1)
        group_attr = dataset.protected_attribute_names[0]
        group_ind = dataset.feature_names.index(group_attr)
        group_vals = list(set(X_changed[:,group_ind]))

        distances = np.zeros(self.no_neighbors)
        indices = np.zeros(self.no_neighbors)
        for x,i in zip(X_changed, range(len(X_changed))):
            cost = dataset.dynamic_cost(dataset.features, np.repeat([x],len(dataset.features), axis=0))
            dists = np.float_power(cost, -1)
            #print("Weights:", dists)
            #for k,d in zip(range(len(dists)), dists):
            #    if not (X[k,group_ind] == X_changed[i,group_ind]):
            #        assert(np.isclose(d,0.0))
            #print(self.no_neighbors)
            indices = np.argpartition(dists, len(dists)-self.no_neighbors)[-self.no_neighbors:]

            #print("indices:", indices)
            distances = dists[indices]
            acc = {key: 0 for key in set(dataset.labels.ravel())}
            result = reduce(lambda x,y: {**x, y[1]: x[y[1]] + y[0]}, np.array(list(zip(dists, dataset.labels.ravel())))[indices], acc)
            label,_ = sorted(result.items(), key=lambda x: x[1], reverse=True)[0]
            Y_changed[i] = [label]

        unique = np.unique(X_changed, axis=0)


#        for group_val in group_vals:
#            # fit KNN to unchanged (during simulation) datapoints
#            cost_dist = lambda x,y: dataset.dynamic_cost(np.array([y]),np.array([x]))[0]
#
#            group_mask = dataset.features[:,group_ind] == group_val
#            nbrs = KNeighborsClassifier(algorithm='brute',n_neighbors=self.no_neighbors, weights='distance', metric=cost_dist, n_jobs=1).fit(dataset.features[group_mask], dataset.labels.ravel()[group_mask])
#
#            ch_group_mask = np.where(X_changed[:,group_ind] == group_val)[0]
#            labels = nbrs.predict(X_changed[ch_group_mask])
#            Y_changed[ch_group_mask] = list(map(lambda x: [x],labels))

        Y[changed_indices] = Y_changed

#        assert(Y.sum() >= dataset.labels.sum())
        # update labels (ground truth)
        dataset_.features = X
        dataset_.labels = np.array(Y.tolist())
        #print(dataset_.labels.sum(), " before: ", dataset.labels.sum())
        return dataset_

    def transform(self, dataset):
        dataset_ = self._do_simulation(dataset)

        # create df for incentives
        ft_names_orig = list(map(lambda x: x+"_orig", dataset.feature_names))

        self.incentive_df = pd.DataFrame(data=self.incentives, columns=['uid'] + ft_names_orig + dataset.feature_names + ['incentive'], dtype=float)
        self.incentives=[]


        return dataset_


