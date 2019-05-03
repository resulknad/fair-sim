import numpy as np
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from aif360.datasets import BinaryLabelDataset
from plot import _df_selection, count_df

import re

from transformer import AgentTransformer

class Simulation(object):
    def __init__(self, dataset, AgentCl, learner, cost_distribution, split=[0.5], collect_incentive_data=False, no_neighbors=51, avg_out_incentive=1, cost_distribution_dep=None):
        self.dataset = dataset
        self.no_neighbors = no_neighbors
        self.cost_distribution = cost_distribution
        self.learner = learner
        self.split = split
        self.AgentCl = AgentCl
        self.avg_out_incentive = avg_out_incentive
        self.collect_incentive_data = collect_incentive_data
        self.cost_distribution_dep = cost_distribution_dep

    def no_classes(self, dataset):
        return len(set(dataset.labels.ravel()))

    @staticmethod
    def dataset_from_matrix(x, dataset):
        df = pd.DataFrame(data=x, columns=dataset.feature_names + dataset.label_names)
        dataset_ = BinaryLabelDataset(df=df, label_names=dataset.label_names, protected_attribute_names=dataset.protected_attribute_names)

        dataset_ = dataset.align_datasets(dataset_)
        #dataset_.favorable_label = dataset.favorable_label
        dataset_.validate_dataset()
        return dataset_


    def start_simulation(self, runs=1,scale=True):
        res_list = []
        for i in range(runs):
            res_list.append(self._simulate(scale))
        return SimulationResultSet(res_list,runs=runs)




    def _simulate(self, scale):
        self.scaler = MaxAbsScaler()
        dataset = self.dataset.copy(deepcopy=True)
        # we need at least one example for each class in each of the two splits
        while True:
            train,test = dataset.split(self.split, shuffle=True)
            break
            if self.no_classes(train) >= 2 and self.no_classes(test) >= 2:
                break
        train_indices = list(map(int, train.instance_names))
        test_indices = list(map(int, test.instance_names))
        print(train.features.shape, test.features.shape)


        self.train, self.test = train,test
        if scale:
            train.features = self.scaler.fit_transform(train.features)
            test.features = self.scaler.transform(test.features)
            dataset.features = self.scaler.transform(dataset.features)

        dataset.infer_domain()


        #t = dataset.features
        #rows = np.where(t[:,1] == 0.0)
        #result = t

        # learner moves
        h = self.learner.fit(train)

        ft_names = dataset.protected_attribute_names
        ft_indices = list(map(lambda x: not x in ft_names, dataset.feature_names))

        self.Y_predicted = h(dataset.features, False)

        # agents move
        at = AgentTransformer(self.AgentCl, h, self.cost_distribution, self.scaler, collect_incentive_data=self.collect_incentive_data, no_neighbors=self.no_neighbors, avg_out_incentive=self.avg_out_incentive, cost_distribution_dep=self.cost_distribution_dep)

        dataset_ = at.transform(dataset)
        train_ = Simulation.dataset_from_matrix(np.hstack((dataset_.features[train_indices,:], dataset_.labels[train_indices])),dataset)
        test_ = Simulation.dataset_from_matrix(np.hstack((dataset_.features[test_indices,:], dataset_.labels[test_indices])),dataset)

        acc_h = self.learner.accuracy(test)


        # update changed features

        #dataset_ = dataset_from_matrix(np.hstack((np.vstack((train_.features, test_.features)), np.vstack((train_.labels, test_.labels)))), dataset)
        self.Y_new_predicted = h(dataset_.features, False)

        acc_h_post = self.learner.accuracy(test_)
        #print("Accuracy (h) post",acc_h_post)

        # fit data again, see if accuracy changes
        self.learner.fit(train_)
        acc_h_star_post = self.learner.accuracy(test_)
        #print("y=1",sum((train_.labels + test_.labels).ravel())," <- ", sum((train.labels + test.labels).ravel()))



        # construct datasets for features
        # including predicted label
        dataset.features = self.scaler.inverse_transform(dataset.features)
        dataset_df = dataset.convert_to_dataframe(de_dummy_code=True)[0]
        dataset_df['credit_h'] = pd.Series(self.Y_predicted, index=dataset_df.index)

        dataset_.features = self.scaler.inverse_transform(dataset_.features)
        dataset_new_df = dataset_.convert_to_dataframe(de_dummy_code=True)[0]
        dataset_new_df['credit_h'] = pd.Series(self.Y_new_predicted, index=dataset_new_df.index)

        res = SimulationResult()
        res.df = dataset_df
        res.df_new = dataset_new_df
        res.eps = abs(acc_h_star_post-acc_h_post)
        res.acc_h = acc_h
        res.acc_h_post = acc_h_post
        res.acc_h_star_post = acc_h_star_post
        res.incentives = at.incentive_df
        return res

class SimulationResultSet:
    results = []

    def __init__(self, results, runs=0):
        self.results = results
        self.runs = runs
        self._average_vals()

    def _avg_incentive(self, feature, group):
        #print(group)
        combined = pd.concat(list(map(lambda x: x.incentives[[group, feature, 'incentive']], self.results)))
        return combined.groupby([group, feature]).mean()


    def _average_vals(self):
        self.eps = np.average(list(map(lambda x: x.eps, self.results)))
        #print(list(map(lambda x: x.eps, self.results)))
        self.eps_std = np.std(list(map(lambda x: x.eps, self.results)))
        self.acc_h = np.average(list(map(lambda x: x.acc_h, self.results)))
        self.acc_h_std = np.std(list(map(lambda x: x.acc_h, self.results)))
        #self.acc_h = acc_h
        #self.acc_h_post = acc_h_post
        #self.acc_h_star_post = acc_h_star_post
        #self.incentives = at.incentive_df

    def _pr(self, group, time='post', ft_name='credit_h'):
        pr_list = []

        for res in self.results:
            df = res.df_new if time == 'post' else res.df
            count = count_df(df, [{ft_name: 1, **group}, {ft_name: 0, **group}])
            total = count.sum()
            #print(count,total)
            pr, _ = count / total
            pr_list.append(pr)
        return np.mean(pr_list)

    # returns average of stat parity diff post sim
    def stat_parity_diff(self, unpriv, priv):
        #up_pr = self._pr(unpriv,'pre', 'y')
        #p_pr = self._pr(priv,'pre', 'y')
        #print('(pre, y) up',up_pr,', p', p_pr)

        #up_pr = self._pr(unpriv, 'pre')
        #p_pr = self._pr(priv, 'pre')
        #print('(pre) up',up_pr,', p', p_pr)

        #up_pr = self._pr(unpriv,'post', 'y')
        #p_pr = self._pr(priv,'post', 'y')
        #print('(post, y) up',up_pr,', p', p_pr)

        up_pr = self._pr(unpriv)
        p_pr = self._pr(priv)
        #print('(post, h) up',up_pr,', p', p_pr)

        return (up_pr-p_pr)

    def tpr(self, selection_criteria={}, truth_ft='y', pred_ft='credit_h', time='post'):
        dfs = list(map(lambda r: r.df_new if time == 'post' else r.df, self.results))

        crit_true = {**selection_criteria, truth_ft: 1}
        crit_true_pos = {**selection_criteria, truth_ft: 1, pred_ft: 1}

        n_true = list(map(lambda df: count_df(df, [crit_true]), dfs))
        n_true_pos = list(map(lambda df: count_df(df, [crit_true_pos]), dfs))

        tprs = np.divide(n_true_pos, n_true)
        return np.mean(tprs), np.std(tprs)


    def feature_average(self, feature, selection_criteria={}):
        ft_values = list(reduce(lambda x,y: np.hstack((x,y)), map(lambda x: list(_df_selection(x.df, selection_criteria)[feature]), self.results)))
        ft_means = list(map(lambda x: np.mean(list(_df_selection(x.df, selection_criteria)[feature])), self.results))
        ft_new_values = list(reduce(lambda x,y: np.hstack((x,y)), map(lambda x: list(_df_selection(x.df_new, selection_criteria)[feature]), self.results)))
        ft_new_means = list(map(lambda x: np.mean(list(_df_selection(x.df_new, selection_criteria)[feature])), self.results))

        return np.mean(ft_values),np.std(ft_means),np.mean(ft_new_values), np.std(ft_new_means)
        #df = _df_selection(self.df, selection_criteria)
        #df_new = _df_selection(self.df_new, selection_criteria)

    def __str__(self):
        return ' '.join(("Runs: ", str(self.runs), "\n",
            "Eps: ",str(round(self.eps,2))," (+- ",str(round(self.eps_std,2)),")", "\n",
            "Acc h: ",str(round(self.acc_h,2))," (+- ",str(round(self.acc_h_std,2)),")", "\n"))




class SimulationResult:
    df = {}
    df_new = {}
    incentives = {}
    eps = 0
    acc_h = 0
    acc_h_post = 0
    acc_h_star_post = 0

    # goal: reproduce results on simple set
    # with multiple tries and confidence interval
    # then move on to stat. parity implementation from aif360
    # then do in processing statistical parity
    # then pre processing statistical parity



    def __str__(self):
        attrs = vars(self)
        return "\n".join("\n%s\n %s" % item for item in attrs.items())
        print("Train: ",train.features.shape,", Test: ", test.features.shape)

    @staticmethod
    def average_results(sim_res):
        avg = lambda x: (np.mean(x), np.std(x))
        avg_res = SimulationResult()
        avg_res.acc_h = avg(list(map(lambda x: x.acc_h, sim_res)))
        avg_res.eps = avg(list(map(lambda x: x.eps, sim_res)))

