import itertools
import warnings
import pandas as pd
import numpy as np

from aif360.datasets import GermanDataset
from aif360.datasets import StructuredDataset
from aif360.datasets import BinaryLabelDataset
from scipy.stats import percentileofscore

class SimMixin:
    """
    Mixin, extending AIF360s StructuredDataset with functionalities necessary for the simulation

    :param mutable_features: List of feature names which are mutable
    :param domains: Dict specifying the domain for each feature
    :param cost_fns: Dict specifying cost functions (lambda) for each feature.
    :param discrete: List of discrete feature names.
    """
    def __init__(self, mutable_features=[], domains={}, cost_fns={}, discrete=[]):
        # assert(len(mutable_features)==len(domains)==len(cost_fns)==len(discrete))
        self.mutable_features = mutable_features
        self.domains = domains
        self.cost_fns = cost_fns
        self.discrete = discrete

    def set_cost_fns(self, cost_fns):
        self.cost_fns = cost_fns

    def infer_domain(self):
        """
        Automatically infers domain for features where domain was set to `'auto'`
        """
        # handle domain != list
        self.domains = {k: self._get_domain(k) if type(v) is not list else v for k,v in self.domains.items()}
        self.ranks = self.rank_fns()

    def scale_dummy_coded(self, X):
        """
        Ensures that the values for one dummy-coded feature sum up to 1 (scales accordingly). Called during gradient ascend. You may find an in-depth explanation in the write-up.

        :param X: Feature matrix (dimension `n_instances x n_features`)
        :returns: X' (modified feature matrix)
        """
        #print(np.where(X[:,12]>0.8))

        for k,v in StructuredDataset._parse_feature_names(self.feature_names)[0].items():
            ft_indices = (list(map(lambda x: self.feature_names.index(k + '=' + x), v)))

            #if k == 'property':
            #    print(X[4,ft_indices])

            X[:, ft_indices] = X[:,ft_indices] / X[:, ft_indices].sum(axis=1)[:,None]

            assert(np.isclose(X[:,ft_indices].sum(axis=1).sum(),len(X)))

        return X

    def enforce_dummy_coded(self, X):
        """
        Enforces that for dummycoded features exactly one feature is set to 1, all the others to 0. Called after gradient ascend.

        :param X: Feature matrix (dimension `n_instances x n_features`)
        :returns: X' (modified feature matrix)
        """
        for k,v in StructuredDataset._parse_feature_names(self.feature_names)[0].items():
            ft_indices = (list(map(lambda x: self.feature_names.index(k + '=' + x), v)))
#            print(k,ft_indices, v)
            max_index = np.argmax(X[:,ft_indices], axis=1)

#            for i in range(len(max_index)):
#                if X[i,ft_indices].sum() > 0 and k == 'credit_history':
#                    print(k)
#                    print(X[i,ft_indices])
#                    print((X[i,ft_indices] == 1))

            X[:, ft_indices] = 0
            for i in range(len(max_index)):
                X[i, ft_indices[max_index[i]]] = 1
            for x in X:
                assert(x[ft_indices].sum() == 1)

#        print(X.shape)
        return X

    def rank_fns(self):
        """
        Precomputes the rank functions, which may be used in cost functions. Note that this function creates one rank funktion per protected group and feature.
        """

        X = self.features

        assert(len(self.protected_attribute_names)==1)

        group_attr = self.protected_attribute_names[0]
        group_ind = self.feature_names.index(group_attr)
        group_vals = list(set(X[:,group_ind]))
        ranks = {}
        for group_val in group_vals:
            mask = (X[:,group_ind] == group_val)
            for ft_name, vals in zip(self.feature_names, np.array(self.features).transpose()):
                A = np.array(vals[mask])
                pct = np.array(list(map(lambda x: np.count_nonzero(A <= x)/len(A), A)))

                def create_rank_fn(A, pct, group_val):
                    sorted_interp = np.array(sorted(list(zip(A, pct)), key=lambda x: x[0]))

                    return lambda y: np.interp(y, sorted_interp[:,0], sorted_interp[:,1])

                if ranks.get(group_val, None) is None:
                    ranks[group_val] = {}
                ranks[group_val][ft_name] = create_rank_fn(A, pct, group_val)
        return ranks

    def _is_dummy_coded(self, ft):
        """
        :param ft: Feature name
        :returns: True if ft is dummycoded
        """
        # fix this
        return len(StructuredDataset._parse_feature_names(self.feature_names)[0][ft])

    def vector_to_object(self, vec):
        """
        :param vec: Instance (feature values) in vector form
        :returns: Instance (feature values) in object form (dict)
        """
        return self._dedummy_code_obj({ft: v for ft,v in zip(self.feature_names, vec)})

    def obj_to_vector(self, obj):
        """
        :param obj: Instance (feature values) in object form (dict)
        :returns: Instance (feature values) in vector form
        """
        obj = self._dummy_code_obj(obj)
        return [obj.get(k, None) for k in self.feature_names]

    def _dedummy_code_obj(self, obj, sep='='):
        """
        :param obj: Instance (feature values) in object form (dict)
        :param sep: Seperator used for dummy coding
        :returns: dedummy coded object
        """
        # reimplemented this bc library is too slow for one row only...
        result_obj = obj.copy()
        for k,v in (StructuredDataset._parse_feature_names(self.feature_names)[0]).items():
            # figure out which dummy coded is set to 1
            value_l = list(filter(lambda x: obj[k+sep+x]==1, v))
            value = value_l.pop() if len(value_l)>0 else None

            # convert to non-dummy coded
            result_obj[k] = value

            # remove all dummy coded ie [key=value]
            [result_obj.pop(k + sep + option) for option in v]

        return result_obj

    def _ft_index(self, ft):
        """
        :param ft: Feature name
        :returns: index of feature (position in dataframe)
        """
        return self.feature_names.index(ft)

    def _dummy_code_obj(self, obj, sep='='):
        """
        :param obj: Instance (feature values) in object form (dict)
        :param sep: Seperator used for dummy coding
        :returns: dummy coded object
        """
        result_obj = {}
        for k,v in obj.items():
            if self._is_dummy_coded(k):
                relevant_columns = (filter(lambda x: x.startswith(k + sep), self.feature_names))
                d_coded = {c: 1 if c == k + sep + v else 0 for c in relevant_columns}
                result_obj = {**result_obj, **d_coded}
            else:
                result_obj[k] = v
        return result_obj

    def dynamic_cost(self, X_new, X):
        """
        Returns cost of feature manipulation (as defined in `self.cost_fns`) from `X` to `X_new`.
        :param X_new: Feature matrix (dimension `n_instances x n_features`)
        :param X: Feature matrix (dimension `n_instances x n_features`)
        :returns: List (dimension `n_instances` x 1) with manipulation cost.
        """
       # if len(X_new) == 1:
       #     print(dict(zip(self.feature_names, X[0]))['age'])
       #     print(dict(zip(self.feature_names, X_new[0]))['age'])
        assert(len(self.protected_attribute_names)==1)
        group_attr = self.protected_attribute_names[0]
        group_ind = self.feature_names.index(group_attr)
        group_vals = list(set(X[:,group_ind]))

        cost = np.zeros(len(X), dtype=np.float64)

        # no cost fns defined
        if self.cost_fns is None or len(self.cost_fns) == 0:
            return cost

        for x_new, x, ft in zip(X_new.transpose(), X.transpose(), self.feature_names):
            # cost fns are defined per feature, sum them up
            if ft in self.cost_fns:
                for group_val in group_vals:
                    mask = (X[:,group_ind] == group_val)
                    cost_inc = self.cost_fns[ft](x_new[mask], x[mask], self.ranks[group_val][ft])
                    cost[mask] += cost_inc
        return cost


    def _get_domain(self, ft):
        """
        Infers domain of feature.
        :param ft: Feature name
        :returns: Domain
        """
        if callable(self.domains[ft]):
            return [self.domains[ft]()]
        elif self._is_dummy_coded(ft):
            raise Exception("Can't use dummy coded for sim")
            warnings.warn("Use set of values present in dataset to infer domain for feature " + ft)
            # discrete, dummy coded
            return StructuredDataset._parse_feature_names(self.feature_names)[0][ft]
        elif ft in self.discrete:
            # discrete
            #warnings.warn("Use set of values present in dataset to infer domain for feature " + ft)
            return list(set(self.features[:,self._ft_index(ft)]))
        else:
            # continious
            df, _ = self.convert_to_dataframe()
            warnings.warn("Used min/max for feature " + ft + " to infer domain + unsupported/not implemented yet")
            return (min(df[ft]), max(df[:,ft]))

    def _discrete_and_mutable(self):
        """
        :returns: Discrete and mutable feature names.
        """
        return list(set(self.mutable_features) & set(self.discrete))

    def _continuous_and_mutable(self):
        """
        :returns: Continious and mutable feature names.
        """
        return list(set(self.mutable_features) - set(self.discrete))

    def continuous_domains(self):
        """
        :returns: Continious domains.
        """
        ft_names = self._continuous_domains()
        domains = [self.domains[x] for x in ft_names]

        return (ft_names, domains)

    def discrete_permutations(self):
        """
        :returns: Iterator over all possible feature value combinations for discrete mutable features.
        """
        ft_names = self._discrete_and_mutable()
        domains = [self.domains[x] for x in ft_names]

        crossproduct_iter = itertools.product(*domains)
        return (ft_names,crossproduct_iter)
