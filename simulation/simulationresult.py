import numpy as np
from utils import count_df,_df_selection
from functools import reduce
import pandas as pd

class SimulationResultSet:
    results = []

    def save(self, filename):
        dumpfile = open(filename, 'ab')
        pickle.dump(self, dumpfile)
        dumpfile.close()

    def load(filename):
        rs = {}
        dumpfile = open(filename, 'ab')
        pickle.dump(rs, dumpfile)
        dumpfile.close()
        return rs

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
    def stat_parity_diff(self, unpriv, priv, time='post'):
        up_pr = self._pr(unpriv, time=time)
        p_pr = self._pr(priv, time=time)

        return (p_pr-up_pr)


    def tpr(self, selection_criteria={}, truth_ft='y', pred_ft='credit_h', time='post'):
        dfs = list(map(lambda r: r.df_new if time == 'post' else r.df, self.results))

        crit_true = {**selection_criteria, truth_ft: 1}
        crit_true_pos = {**selection_criteria, truth_ft: 1, pred_ft: 1}

        n_true = list(map(lambda df: count_df(df, [crit_true]), dfs))
        n_true_pos = list(map(lambda df: count_df(df, [crit_true_pos]), dfs))

        tprs = np.divide(n_true_pos, n_true)
        return np.mean(tprs), np.std(tprs)


    def feature_average(self, feature, selection_criteria={}):
        ft_values = np.array(list(reduce(lambda x,y: np.hstack((x,y)), map(lambda x: list(_df_selection(x.df, selection_criteria)[feature]), self.results))))
        if ft_values.dtype == np.float64:
            ft_means = list(map(lambda x: np.mean(list(_df_selection(x.df, selection_criteria)[feature])), self.results))
            ft_new_values = list(reduce(lambda x,y: np.hstack((x,y)), map(lambda x: list(_df_selection(x.df_new, selection_criteria)[feature]), self.results)))
            ft_new_means = list(map(lambda x: np.mean(list(_df_selection(x.df_new, selection_criteria)[feature])), self.results))
            return np.mean(ft_values),np.std(ft_means),np.mean(ft_new_values), np.std(ft_new_means)
        else:
            return 0,0,0,0


        #df = _df_selection(self.df, selection_criteria)
        #df_new = _df_selection(self.df_new, selection_criteria)

    def feature_table(self, selection_criteria=[]):
        data = []
        data.append(count_df(self.results[0].df_new, selection_criteria))
        for ft in list(self.results[0].df_new):
            row = []
            for sc in selection_criteria:
                pre_mean, pre_std, post_mean, post_std = self.feature_average(ft, sc)
                row = row + [str(round(pre_mean,2)) + " -> " + str(round(post_mean,2))]
            data.append(row)

        return pd.DataFrame(data=data, index=["N"] + list(self.results[0].df_new))

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

