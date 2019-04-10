import itertools
import warnings
import pandas as pd
import numpy as np

from aif360.datasets import GermanDataset
from aif360.datasets import StructuredDataset


class SimMixin:
    def __init__(self, mutable_features=[], domains={}, cost_fns={}, discrete=[]):
        # assert(len(mutable_features)==len(domains)==len(cost_fns)==len(discrete))
        self.mutable_features = mutable_features
        self.domains = domains
        self.cost_fns = cost_fns
        self.discrete = discrete

        # handle domain = 'auto'
        self.domains = {k: self._get_domain(k) if v == 'auto' else v for k,v in domains.items()}
        print(self.domains)

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


    def _get_domain(self, ft):
        if self._is_dummy_coded(ft):
            # discrete
            return StructuredDataset._parse_feature_names(self.feature_names)[0][ft]
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
    'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}],
    'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
                                 {1.0: 'Old', 0.0: 'Young'}],
}

def custom_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['personal_status'].replace(status_map)
    df['credit'] = df['credit'].map(lambda x: 2-x)
    return df

class GermanSimDataset(GermanDataset, SimMixin):
    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        kwargs['custom_preprocessing'] = custom_preprocessing
        kwargs['metadata'] = default_mappings
        GermanDataset.__init__(*(tuple([self]) + args), **kwargs)
        SimMixin.__init__(self, **sim_args)


