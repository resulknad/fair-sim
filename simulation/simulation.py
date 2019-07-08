from .simulationresult import SimulationResultSet, SimulationResult
from transformer import AgentTransformer
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from aif360.datasets import BinaryLabelDataset
import numpy as np
import pandas as pd

class Simulation(object):
    """Class glueing all the pieces together. Performs whole simulation.

    :param dataset: Dataset which extends :py:obj:`mutabledataset.SimMixin`
    :param AgentCl: Class defining agent behavior, namely `benefit` and `cost`.
    :param learner: Class defining learner behavior, namely `fit` and `predict`.
    :param split: Defines portion used for fitting the learner. Rest is used for determining `eps` value, regarding the epsilon equilibrium. Simulation is done on the whole dataset.
    :param cost_distribution: Passed on to AgentTransformer.
    :param cost_distribution_dep: Passed on to AgentTransformer.
    :param no_neighbors: Passed on to AgentTransformer.
    :param max_it: Passed on to AgentTransformer.
    :param collect_incentive_data: Passed on to AgentTransformer.
    """
    def __init__(self, dataset, AgentCl, learner, cost_distribution, split=[0.5], collect_incentive_data=False, no_neighbors=60, cost_distribution_dep=None, max_it=130):
        self.dataset = dataset
        self.no_neighbors = no_neighbors
        self.cost_distribution = cost_distribution
        self.max_it = max_it
        self.learner = learner
        self.split = split
        self.AgentCl = AgentCl
        self.collect_incentive_data = collect_incentive_data
        self.cost_distribution_dep = cost_distribution_dep

    def no_classes(self, dataset):
        """
        :param dataset: Some AIF360 dataset
        :returns: Number of distinct labels (classes)
        """
        return len(set(dataset.labels.ravel()))

    def start_simulation(self, runs=1,scale=True):
        """
        :param runs: Run simulation multiple times with the same parameters
        :param scale: Perform scaling on dataset features.
        :returns: Modified dataset including new ground truth labels
        :rtype: :py:obj:`simulation.SimulationResultSet`
        """
        res_list = []
        for i in range(runs):
            res_list.append(self._simulate(scale))
        return SimulationResultSet(res_list,runs=runs)

    def _simulate(self, scale):
        """
        Private entrypoint to perform a single simulation

        :param scale: Perform scaling on dataset features
        :returns: Modified dataset including new ground truth labels
        :rtype: :py:obj:`simulation.SimulationResult`
        """
        self.scaler = MaxAbsScaler()
        dataset = self.dataset.copy(deepcopy=True)
        # we need at least one example for each class in each of the two splits
        while True:
            train,test = dataset.split(self.split, shuffle=False)
            break
            if self.no_classes(train) >= 2 and self.no_classes(test) >= 2:
                break
        train_indices = list(map(int, train.instance_names))
        test_indices = list(map(int, test.instance_names))


        self.train, self.test = train,test
        if scale:
            train.features = self.scaler.fit_transform(train.features)
            test.features = self.scaler.transform(test.features)
            dataset.features = self.scaler.transform(dataset.features)

        dataset.infer_domain()

        # learner moves
        self.learner.fit(train)

        ft_names = dataset.protected_attribute_names
        ft_indices = list(map(lambda x: not x in ft_names, dataset.feature_names))

        self.Y_predicted = self.learner.predict(dataset.features)
        self.Y_predicted_pr = self.learner.predict_proba(dataset.features)

        # agents move
        at = AgentTransformer(self.AgentCl, self.learner, self.cost_distribution, collect_incentive_data=self.collect_incentive_data, no_neighbors=self.no_neighbors, cost_distribution_dep=self.cost_distribution_dep, max_it=self.max_it)

        dataset_ = at.transform(dataset)

        train_ = utils.dataset_from_matrix(np.hstack((dataset_.features[train_indices,:], dataset_.labels[train_indices])),dataset)
        test_ = utils.dataset_from_matrix(np.hstack((dataset_.features[test_indices,:], dataset_.labels[test_indices])),dataset)

        acc_h = self.learner.accuracy(test)


        # update changed features

        #dataset_ = dataset_from_matrix(np.hstack((np.vstack((train_.features, test_.features)), np.vstack((train_.labels, test_.labels)))), dataset)
        self.Y_new_predicted = self.learner.predict(dataset_.features)
        self.Y_new_predicted_pr = self.learner.predict_proba(dataset_.features)

        acc_h_post = self.learner.accuracy(test_)

        # fit data again, see if accuracy changes
        self.learner.fit(train_)
        acc_h_star_post = self.learner.accuracy(test_)


        # construct datasets for features
        # including predicted label
        if scale:
            dataset.features = self.scaler.inverse_transform(dataset.features)
        dataset_df = dataset.convert_to_dataframe(de_dummy_code=True)[0]
        dataset_df['credit_h'] = pd.Series(self.Y_predicted, index=dataset_df.index)
        dataset_df['credit_h_pr'] = pd.Series(self.Y_predicted_pr, index=dataset_df.index)
        if scale:
            dataset_.features = self.scaler.inverse_transform(dataset_.features)
        dataset_new_df = dataset_.convert_to_dataframe(de_dummy_code=True)[0]
        dataset_new_df['credit_h'] = pd.Series(self.Y_new_predicted, index=dataset_new_df.index)
        dataset_new_df['credit_h_pr'] = pd.Series(self.Y_new_predicted_pr, index=dataset_new_df.index)

        res = SimulationResult()
        res.df = dataset_df
        res.df_new = dataset_new_df
        res.eps = abs(acc_h_star_post-acc_h_post)
        res.acc_h = acc_h
        res.acc_h_post = acc_h_post
        res.acc_h_star_post = acc_h_star_post
        res.incentives = at.incentives
        return res
