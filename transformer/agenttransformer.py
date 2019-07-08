
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import starmap
from functools import reduce
import time
import traceback

from aif360.algorithms import Transformer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import uuid

# copied from SCIKIT OPTIMIZE and modified:

def approx_fprime(xk, f, epsilon, args=(), f0=None, immutable=[]):
    """Based on `approx_fprime` from `scikit.optimize`. Approximates the gradient of a multivariate function using finite central difference approximation at a certain point `xk`.

        :param xk: Points of evaluation. Matrix with dimension `n_instances x n_features`.
        :param f: Function whose gradient we want to approximate.
        :param epsilon: Step size (`h`) for approximation
        :param immutable: Feature index where gradient approximation is not performed.
        :returns: Post-simulation dataset with updated feature vectors and updated ground truth.
    """
    if f0 is None:
        f0 = f(*((xk,) + args))
    n_instances, n_features = xk.shape
    grad = np.zeros(xk.shape, float)
    ei = np.zeros(xk.shape, float)
    for k in range(n_features):
        if k not in immutable:
            # approx gradient for all n_instances for feature k
            ei[:,k] = 1.0
            d = epsilon * ei
            grad[:,k] = (f(*((xk + d*0.5,) + args)) - f(*((xk - d*0.5,) + args))) / d[0][k]
            ei[:,k] = 0.0

    return grad, f0

class AgentTransformer(Transformer):
    """Performs manipulation of feature vectors in response to hypothesis from learner.

    :param agent_class: Class defining benefit and cost functions
    :param h: Learners best response to dataset.
    :param cost_distribution: Distribution function for fixed cost (indpendent of features), e.g. lambda size: np.random.normal(mu, sigma, size)
    :param cost_distribution_dep: Distribution function for fixed cost (dependent on features), e.g. lambda x: 1 if x[0] == 'black' else 0, parameter is simply the feature vector of one instance
    :param cost_distribution_dep: Distribution function for fixed cost (dependent on features), e.g. lambda x: 1 if x[0] == 'black' else 0, parameter is simply the feature vector of one instance
    :param no_neighbors: Number of neighbors to consider in KNN for ground truth update
    :param collect_incentive_data: Collect debugging information during gradient ascend
    :param max_it: Maximum iterations for gradient ascend

    """
# this class is not thread safe
    def __init__(self, agent_class, h, cost_distribution, no_neighbors=51, collect_incentive_data=False, cost_distribution_dep=None, max_it=130):
        self.agent_class = agent_class
        self.h = h
        self.cost_distribution = cost_distribution
        self.no_neighbors = no_neighbors
        self.max_it = max_it
        self.collect_incentive_data = collect_incentive_data
        self.cost_distribution_dep = cost_distribution_dep
        self.incentives = []

        super(AgentTransformer, self).__init__(
            agent_class=agent_class,
            h=h,
            cost_distribution=cost_distribution)

    def _optimal_x_gd(self, dataset, cost):
        """Performs gradient ascend on the incentive function specified in :py:attr:`agent_class`

        :param dataset: Some dataset, which extends `SimMixin`.
        :param cost: Cost vector for fixed cost. Dimension should match dataset `n_instances x 1`.

        :return: Tuple consisting of manipulated features and the incentive value corresponding to the manipulations
        """

        X = dataset.features
        y = dataset.labels

        assert(len(dataset.protected_attribute_names)==1)
        immutable = [dataset.feature_names.index(dataset.protected_attribute_names[0])]

        # x0
        a = self.agent_class(self.h, dataset, cost, np.copy(X), y)
        a.threshold = self.h.threshold

        eps = 0.05
        gradient,incentive = approx_fprime(X, a.incentive, eps, immutable=immutable)

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

        def collect_incentive_data(X):
            if self.collect_incentive_data:
                self.incentives.append({'features': X,
                    'benefit': a.benefit(X),
                    'cost': a.cost(X),
                    'boost': 0.,
                    'names': dataset.feature_names})

        collect_incentive_data(X)

        print("starting GA")
        incentive_last = 0
        MAXIT = self.max_it
        for i in range(MAXIT):
            incentive_last = incentive

            X = np.add(X,0.1* gradient)
            X = dataset.scale_dummy_coded(X)
            X = np.clip(X, min_domain, max_domain)

            gradient, incentive = approx_fprime(X, a.incentive, eps, immutable=immutable)
            collect_incentive_data(X)

            if (abs(incentive - incentive_last) < 0.001).all() and i>20:
                break


        print("Gradient ascend stopped after ",i+1,"iterations (max i", MAXIT,", min 20)")

        X = dataset.enforce_dummy_coded(X)

        # finds closest neighbor for discrete features
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

        X_ = np.array(X_)
        X_ = X_.transpose()
        collect_incentive_data(X_)

        return a.incentive(X_), X_


    def _do_simulation(self, dataset):
        """Performs simulation on dataset. Calls `_optimal_x_gd` to approximate best-response feature vectors of the agents. Performs KNN for ground truth update.

            :param dataset: Dataset to perform simulation on. Must extend `SimMixin`.
            :returns: Post-simulation dataset with updated feature vectors and updated ground truth.
        """
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
        print("incentive > 0 :", (incentives > 0).sum())
        print("incentive <=0 :", np.mean(incentives[incentives <= 0]))

        # only update feature vector for agents with incentive >0
        for x_vec,x,y,c,incentive in zip(X,dataset.features, dataset.labels, cost, incentives):
            if incentive > 0 and not (x_vec == x).all():
                features_.append(np.array(x_vec))
                changed_indices.append(i)
            else:
                features_.append(np.array(x))
            labels_.append(y)
            i+=1
        dataset_.features = features_

        X = np.array(features_)
        Y = np.array(labels_)

        # no changes during simulation
        # no need to assign new labels with KNN
        if len(changed_indices) == 0:
            dataset_.features = X
            dataset_.labels = dataset.labels #np.array(Y.tolist())
            return dataset_

        unchanged_indices = np.setdiff1d(list(range(len(X))), changed_indices)
        X_changed = X[changed_indices,:]
        Y_changed = Y[changed_indices]

        X_unchanged = X[unchanged_indices,:]
        Y_unchanged = Y[unchanged_indices]

        assert(len(X_changed)==len(changed_indices))
        assert(len(X_unchanged)==len(X)-len(changed_indices))


        # KNN
        assert(len(dataset.protected_attribute_names)==1)
        group_attr = dataset.protected_attribute_names[0]
        group_ind = dataset.feature_names.index(group_attr)
        group_vals = list(set(X_changed[:,group_ind]))

        distances = np.zeros(self.no_neighbors)
        indices = np.zeros(self.no_neighbors)
        for x,i in zip(X_changed, range(len(X_changed))):
            cost = dataset.dynamic_cost(dataset.features, np.repeat([x],len(dataset.features), axis=0))
            cost[cost==0] = 0.00000000000000000000001
            dists = np.float_power(cost, -1)
            indices = np.argpartition(dists, len(dists)-self.no_neighbors)[-self.no_neighbors:]

            #print("indices:", indices)
            distances = dists[indices]
            acc = {key: 0 for key in set(dataset.labels.ravel())}
            result = reduce(lambda x,y: {**x, y[1]: x[y[1]] + y[0]}, np.array(list(zip(dists, dataset.labels.ravel())))[indices], acc)
            label,_ = sorted(result.items(), key=lambda x: x[1], reverse=True)[0]
            Y_changed[i] = [label]

        unique = np.unique(X_changed, axis=0)


        Y[changed_indices] = Y_changed


        # update labels (ground truth)
        dataset_.features = X
        dataset_.labels = np.array(Y.tolist())

        #print(dataset_.labels.sum(), " before: ", dataset.labels.sum())
        return dataset_

    def transform(self, dataset):
        dataset_ = self._do_simulation(dataset)

        return dataset_
