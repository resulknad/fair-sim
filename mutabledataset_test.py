import unittest
from aif360.datasets import BinaryLabelDataset

from mutabledataset import SimMixin, GermanSimDataset
from learner import _accuracy
from simulation import Simulation
import pandas as pd
import numpy as np

class TestLearner(object):
    def fit(self, dataset):
        def h(x, single=True):
            res = list(map(lambda x: int(x[0]<1), x))
            return res[0] if single else res
        self.h = h
        return h

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

class TestAgent:
    def __init__(self, *args, **kwargs):
        return

    async def incentive(self, x_new):
        return 0

    async def cost(self, x_new):
        return 0

    async def benefit(self, x_new):
        return 0

class TestDataset(BinaryLabelDataset, SimMixin):
    def _generateData(self):
        data = [[0.5,0], [0.5,0], [1,1], [1,1]]
        return pd.DataFrame(data=data, columns=['x', 'y'])

    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        df = self._generateData()

        kwargs = {**kwargs, 'df':df, 'label_names':['y']}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)


class TestSimulation(unittest.TestCase):
    def test_rank_fns(self):
        dataset = TestDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={}, protected_attribute_names=[])
        dataset.infer_domain()
        fns = dataset.rank_fns()

        self.assertEqual(list(map(fns['x'], np.linspace(0,1,10))), [0.0, 0.0, 0.0, 0.0, 0.0, .5, .5, .5, .5, 1.0])

if __name__ == '__main__':
    unittest.main()
