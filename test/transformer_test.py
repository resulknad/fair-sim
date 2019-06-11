import unittest
from aif360.datasets import BinaryLabelDataset
from mutabledataset import SimMixin
from transformer import AgentTransformer
from agent import RationalAgent
import numpy as np
import pandas as pd

class TestGroupDataset(BinaryLabelDataset, SimMixin):
    def _generateData(self):

        data = [[1,0,0], [1.9,0,1], [2.1,1,0], [3,1,1]]

        return pd.DataFrame(data=data, columns=['x','group', 'y'])


    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        df = self._generateData()

        kwargs = {**kwargs, 'df':df, 'label_names':['y']}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)

class TestDataset(BinaryLabelDataset, SimMixin):
    def _generateData(self):
        data = [[1,0], [1.5,0], [2.5,1], [3,1]]

        return pd.DataFrame(data=data, columns=['x', 'y'])


    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        df = self._generateData()

        kwargs = {**kwargs, 'df':df, 'label_names':['y']}

        BinaryLabelDataset.__init__(self, **kwargs)
        SimMixin.__init__(self, **sim_args)


class TestLearner:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return np.clip(np.interp(X[:,0], self.X, self.y),None,0.51)

class TestAgentTransformer(unittest.TestCase):

    def test_zero_fixed_cost(self):
        # this tests whether group is included in KNN (shouldnt with below def of cost fn)
        # and if KNN really takes the closest neighbor
        # and the clipped predict_proba which incentivizes people not to move above the boundary
        for neighbors in [1,2]:
            dataset = TestGroupDataset(mutable_features=['x'],
                domains={'x': 'auto'},
                discrete=['x'],
                cost_fns={
                    'group': lambda x_new, x, rank: np.abs(x_new-x)*np.nan_to_num(float('inf')),
                    'x': lambda x_new, x, _: np.abs(x_new - x)/100.}, protected_attribute_names=['group'])#'x': cost_lambda})
            dataset.infer_domain()

            at = AgentTransformer(RationalAgent, TestLearner([0,2,3],[0,0.51,1]), lambda size: [0]*size, None, no_neighbors=neighbors)

            dataset_ = at.transform(dataset)
            expectedFeatures = [[2.1, 0.], [2.1, 0.], [2.1, 1.], [3, 1.]]
            expectedLabels = [1] * 2 + [0,1]
            assert((expectedFeatures == dataset_.features).all())
            assert((expectedLabels == dataset_.labels.ravel()).all())


if __name__ == '__main__':
    unittest.main()
