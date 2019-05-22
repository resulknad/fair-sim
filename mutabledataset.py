import itertools
import warnings
import pandas as pd
import numpy as np

from aif360.datasets import GermanDataset
from aif360.datasets import StructuredDataset
from aif360.datasets import BinaryLabelDataset
from scipy.stats import percentileofscore


class SimMixin:
    def __init__(self, mutable_features=[], domains={}, cost_fns={}, discrete=[]):
        # assert(len(mutable_features)==len(domains)==len(cost_fns)==len(discrete))
        self.mutable_features = mutable_features
        self.domains = domains
        self.cost_fns = cost_fns
        self.discrete = discrete

    def set_cost_fns(self, cost_fns):
        self.cost_fns = cost_fns

    def infer_domain(self):
        # handle domain != list
        self.domains = {k: self._get_domain(k) if type(v) is not list else v for k,v in self.domains.items()}
        self.ranks = self.rank_fns()

    def scale_dummy_coded(self, X):

        #print(np.where(X[:,12]>0.8))

        for k,v in StructuredDataset._parse_feature_names(self.feature_names)[0].items():
            ft_indices = (list(map(lambda x: self.feature_names.index(k + '=' + x), v)))

            #if k == 'property':
            #    print(X[4,ft_indices])

            X[:, ft_indices] = X[:,ft_indices] / X[:, ft_indices].sum(axis=1)[:,None]

            assert(np.isclose(X[:,ft_indices].sum(axis=1).sum(),len(X)))

        return X

    def enforce_dummy_coded(self, X):
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
        # fix this
        return len(StructuredDataset._parse_feature_names(self.feature_names)[0][ft])

    def vector_to_object(self, vec):
        return self._dedummy_code_obj({ft: v for ft,v in zip(self.feature_names, vec)})

    def obj_to_vector(self, obj):
        obj = self._dummy_code_obj(obj)
        return [obj.get(k, None) for k in self.feature_names]

    def _dedummy_code_obj(self, obj, sep='='):
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
        return self.feature_names.index(ft)

    def _dummy_code_obj(self, obj, sep='='):
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
        return list(set(self.mutable_features) & set(self.discrete))

    def _continuous_and_mutable(self):
        return list(set(self.mutable_features) - set(self.discrete))

    def continuous_domains(self):
        ft_names = self._continuous_domains()
        domains = [self.domains[x] for x in ft_names]

        return (ft_names, domains)

    def discrete_permutations(self):
        ft_names = self._discrete_and_mutable()
        domains = [self.domains[x] for x in ft_names]

        crossproduct_iter = itertools.product(*domains)
        return (ft_names,crossproduct_iter)

default_mappings = {
    #'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}]#,
    #'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
    #                             {1.0: 'Old', 0.0: 'Young'}],
}

def custom_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['personal_status'].replace(status_map)
    df['foreign_worker'] = df['foreign_worker'].replace({'A201':0, 'A202':1})
    df['savings'] = df['savings'].replace({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65':5})
    df['credit_amount'] = (df['credit_amount']/max(df['credit_amount'])).apply(lambda x: round(x*8)/8.)
    df['has_checking_account'] = df['month'].apply(lambda x: int(not x=='A14'))
    df['status'] = df['status'].replace({'A11': 0, 'A12': 0.5, 'A13': 1, 'A14':0})
    df['month'] = (df['month']/max(df['month'])).apply(lambda x: round(x*8)/8.)
    df['credit'] = df['credit'].map(lambda x: 2-x)
    return df

class GermanSimDataset(GermanDataset, SimMixin):
    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        kwargs['custom_preprocessing'] = custom_preprocessing
        kwargs['metadata'] = default_mappings
        kwargs['categorical_features'] = ['credit_history', 'purpose',
                     'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone'
]

        self.human_readable_labels ={"A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
            "A30": "no credits taken",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account",
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1  <= ... < 4 years",
            "A74": "4  <= ... < 7 years",
            "A75": ">= 7 years",
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
            "A121": "real estate",
            "A122": "building society savings agreement/life insurance",
            "A123": "car or other",
            "A124": "unknown / no property",
            "A141": "bank",
            "A142": "stores",
            "A143": "none",
            "A151": "rent",
            "A152": "own",
            "A153": "for free",
            "A171": "unemployed/ unskilled  - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/ Highly qualified employee/ officer",
            "A191": "none",
            "A192": "yes, registered under the customers name"}

        GermanDataset.__init__(*(tuple([self]) + args), **kwargs)
        SimMixin.__init__(self, **sim_args)


class NonLinSeparableDataset(BinaryLabelDataset, SimMixin):
    def __init__(self, *args, **kwargs):
        n = 100
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        df = pd.DataFrame(data=np.array([[1,0,1],[0,1,1],[1,1,0],[0,1,0],[1,0,0],[0,1,0],[1,1,1],[0,1,1]]*n),columns = ['x0','x1', 'y'])
        kwargs = {'df':df, 'label_names':['y'], 'protected_attribute_names':[]}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)


class CoateLouryDataset(BinaryLabelDataset, SimMixin):
    def __init__(self, *args, **kwargs):
        n = 1
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        f_u_draw = kwargs.pop('f_u_draw')
        f_q_draw = kwargs.pop('f_q_draw')

        self.f_u_draw, self.f_q_draw = f_u_draw, f_q_draw

        # one of each group, both unqualified
        agents = [[0, 0, 0],[1, 0, 0]] * n

        # add their signal theta drawn from f_u
        agents = list(map(lambda x: x + [f_u_draw()], agents))
        df = pd.DataFrame(data=np.array(agents),columns = ['group','educated', 'y', 'signal'])

        kwargs = {'df':df, 'label_names':['y'], 'protected_attribute_names':['group']}

        print(agents)

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)


class SimpleDataset(BinaryLabelDataset, SimMixin):
    def _generateData(self, means, N, threshold):
        def generateX(grp, loc):
            x = np.random.normal(loc=loc, scale=5, size=N)
            x_noisy = x + np.random.normal(loc=0, scale=20, size=N)

            y = list(map(lambda x: 1 if x>threshold else 0, x_noisy))

            x = list(map(round, x))

            return np.vstack(([x],[[grp]*N],[y])).transpose()
        print(means)
        X = np.vstack((generateX(1,means[1]), generateX(0,means[0])))
        return pd.DataFrame(data=X, columns=['x', 'group', 'y'])

    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}
        self.means = kwargs.pop('means', [45,60])
        self.N = kwargs.pop('N', 1000)
        self.threshold = kwargs.pop('threshold', 55)

        df = self._generateData(means=self.means, N=self.N, threshold=self.threshold)

        kwargs = {'df':df, 'label_names':['y'], 'protected_attribute_names':['group']}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)
