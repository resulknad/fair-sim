import unittest
from aif360.datasets import BinaryLabelDataset
from mutabledataset import SimMixin
from transformer import AgentTransformer
from agent import RationalAgent
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


class TestAgentTransformer(unittest.TestCase):
#    def setUp(self):
# construct a simulatable dataset
# use rational agent / override specific methods...
#        print("Set up")

#    def tearDown(self):
#        print("tear down")

    def test_auto_domain(self):
        dataset = TestDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={}, protected_attribute_names=[])#'x': cost_lambda})
        dataset.infer_domain()

        ft, perms = dataset.discrete_permutations()
        self.assertEqual(ft, ['x'])
        self.assertEqual(sorted(list(perms)), sorted([(1.0,),(1.5,),(2.5,),(3.0,)]))

    def test_zero_fixed_cost(self):
        dataset = TestDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={}, protected_attribute_names=[])#'x': cost_lambda})
        dataset.infer_domain()
        h_all = lambda x: list(map(lambda x: 1 if x[0]>2 else 0,x))
        h = lambda x,single=True: h_all(x)[0] if single else h_all(x)
        at = AgentTransformer(RationalAgent, h, lambda size: [0]*size, None, no_neighbors=1)

        dataset_ = at.transform(dataset)
        expectedFeatures = [[2.5], [2.5], [2.5], [2.5]]
        expectedLabels = [1] * 4
        #print("expected", expectedFeatures, "reality", dataset_.features)
        assert((expectedFeatures == dataset_.features).all())
        assert((expectedLabels == dataset_.labels.ravel()).all())

    def test_group_not_included_in_knn(self):
        dataset = TestGroupDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={'x':lambda x_new, x: 0 if abs(abs(x_new-x)-0.2)<0.001 else 1.1}, protected_attribute_names=['group'])

        dataset.infer_domain()
        h = lambda x,single=True: ([1.]*len(x))
        print(h([1,2,3]))
        at = AgentTransformer(RationalAgent, h, lambda size: [0]*size, None, no_neighbors=1)

        dataset_ = at.transform(dataset)
        expectedLabels = [0,0,1,1]
        assert((expectedLabels == dataset_.labels.ravel()).all())

    def test_fixed_cost_too_high(self):
        dataset = TestDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={}, protected_attribute_names=[])#'x': cost_lambda})
        dataset.infer_domain()
        h_all = lambda x: list(map(lambda x: 1 if x[0]>2 else 0,x))
        h = lambda x,single=True: h_all(x)[0] if single else h_all(x)
        at = AgentTransformer(RationalAgent, h, lambda size: [1.1]*size, None, no_neighbors=1)

        dataset_ = at.transform(dataset)

        expectedFeatures = [[1.], [1.5], [2.5], [3.0]]
        expectedLabels = [0]*2 + [1] * 2
        assert((expectedFeatures == dataset_.features).all())
        assert((expectedLabels == dataset_.labels.ravel()).all())

    def test_dynamic_cost(self):
        # should enforce that y=0 manipulate to 1. the others remain
        def cost_fn(x_new, x):
            if x>2:
                return 1
            return 0 if x_new == 1. else 10

        dataset = TestDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={'x': cost_fn}, protected_attribute_names=[])#'x': cost_lambda})
        dataset.infer_domain()

        h_all = lambda x: list(map(lambda x: 1 if x[0]>2 else 0.1,x))
        h = lambda x,single=True: h_all(x)[0] if single else h_all(x)
        at = AgentTransformer(RationalAgent, h, lambda size: [0.0]*size, None, no_neighbors=1)

        dataset_ = at.transform(dataset)

        expectedFeatures = [[1.], [1.], [2.5], [3.0]]
        expectedLabels = [0]*2 + [1] * 2
       # print(at.incentive_df)
        #print(expectedFeatures, "real",dataset_.features)
        assert((expectedFeatures == dataset_.features).all())
        assert((expectedLabels == dataset_.labels.ravel()).all())

    def test_linear_dynamic_cost(self):
        # should enforce that people manipulate towards the boundary (2)
        def cost_fn(x_new, x):
            return x_new/3.

        dataset = TestDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={'x': cost_fn}, protected_attribute_names=[])#'x': cost_lambda})
        dataset.infer_domain()

        h_all = lambda x: list(map(lambda x: 1. if x[0]>2 else 0.,x))
        h = lambda x,single=True: h_all(x)[0] if single else h_all(x)
        at = AgentTransformer(RationalAgent, h, lambda size: [0.0]*size, None, no_neighbors=1)

        dataset_ = at.transform(dataset)

        expectedFeatures = [[2.5], [2.5], [2.5], [2.5]]
        expectedLabels = [1] * 4
        #print("expected", expectedFeatures, "reality", dataset_.features)
        assert((expectedFeatures == dataset_.features).all())
        assert((expectedLabels == dataset_.labels.ravel()).all())

if __name__ == '__main__':
    unittest.main()
