import itertools
import warnings
import pandas as pd
import numpy as np

from aif360.datasets import GermanDataset
from aif360.datasets import StructuredDataset
from aif360.datasets import BinaryLabelDataset
from scipy.stats import percentileofscore
from .simmixin import SimMixin

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
        self.human_readable_labels ={}

        df = self._generateData(means=self.means, N=self.N, threshold=self.threshold)

        kwargs = {'df':df, 'label_names':['y'], 'protected_attribute_names':['group']}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)
