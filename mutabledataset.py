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

    def rank_fns(self):
        ranks = {}
        for ft_name, x in zip(self.feature_names, np.array(self.features).transpose()):
            #x_sorted = sorted(x.copy())
            #x_ranks = rankdata(x_sorted)
            #i = np.searchsorted(y, x_sorted)

            #ranks = df['x'].rank(pct=True)
            #rank = lambda v: ranks[(np.abs(x_scaled - v)).argmin()]
            ranks[ft_name] = lambda y: percentileofscore(x, y, kind='weak')/100.
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

    def dynamic_cost(self, x_new, x):
        # no cost fns defined
        if self.cost_fns is None or len(self.cost_fns) == 0:
            return 0

        assert(len(x_new)==1)
        assert(len(x)==1)
        cost = 0.0

        # cost fns are defined per feature, sum them up
        for new, old,ft in zip(x_new[0],x[0], self.feature_names):
            if ft in self.cost_fns:
                cost += self.cost_fns[ft](new, old, self.ranks[ft])

        #        try:

        #        except Exception:
                    # dirty trick to avoid changing the testcases
                    # TODO!
        #            cost += self.cost_fns[ft](new, old)

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
        self.N = kwargs.pop('N', 100)
        self.threshold = kwargs.pop('threshold', 55)

        df = self._generateData(means=self.means, N=self.N, threshold=self.threshold)

        kwargs = {'df':df, 'label_names':['y'], 'protected_attribute_names':['group']}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)
