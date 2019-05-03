
import nest_asyncio
nest_asyncio.apply()

from mutabledataset import SimpleDataset
from sklearn.preprocessing import MaxAbsScaler
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner
import plot
import numpy as np
import pandas as pd
from learner import StatisticalParityLogisticLearner, StatisticalParityFlipperLogisticLearner
from learner import FairLearnLearner
from learner import RejectOptionsLogisticLearner
from learner import ReweighingLogisticLearner
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import normal
mutable_attr = 'x'
def do_sim(learner, means, threshold):
    #cost_lambda = lambda x_new, x:  x_new/2. + abs(x_new-x)/4.  #1*abs(x_new-x)/4. + 3*pow(x_new, 2.)/4.
    cost_fixed = lambda size: np.abs(np.random.normal(loc=0,scale=0.5,size=size))
    #cost_fixed = lambda size: np.abs(np.random.normal(loc=0.5,scale=0.5,size=size))
    #cost_fixed_dependant = lambda x: np.abs(normal(loc=(0. if x[1] == 1 else 1.25), scale=0.5, size=1)[0])


    g = SimpleDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={'x': None},means=means, N=200, threshold=threshold)

    df = g.convert_to_dataframe()[0]
    x_scaled = MaxAbsScaler().fit_transform(np.array(df['x']).reshape(-1,1))
    ranks = df['x'].rank(pct=True)

    rank = lambda v: ranks[(np.abs(x_scaled - v)).argmin()]

    cost_lambda = lambda x_new, x: 3*abs(rank(x_new)-rank(x))#pow(x_new/2.,2.)+3*abs(x_new-x)/4. #lambda x_new, x: x_new/2.+1*abs(x_new-x)/4.
    g.set_cost_fns({'x': cost_lambda})

    #sns.catplot(x="group", y="x", hue="y", data=g.convert_to_dataframe()[0]);
    #plt.show()

    sim = Simulation(g,
                     RationalAgent,
                     learner,
                     cost_fixed,
                     collect_incentive_data=True,
                     avg_out_incentive=1,
                     no_neighbors=11,
                     split=[0.9])#,
                     #cost_distribution_dep=cost_fixed_dependant)

    result_set = sim.start_simulation(runs=1)
    return result_set

def print_stats(result_set, name):
    print(result_set)
    print("StatPar Δ:", round(result_set.stat_parity_diff({'group': 0}, {'group': 1}),2))
    pre_up_mean, pre_up_std, post_up_mean, post_up_std = tuple(map(lambda x: round(x,2),result_set.feature_average(mutable_attr, {'group':0})))
    pre_p_mean, pre_p_std, post_p_mean, post_p_std = tuple(map(lambda x: round(x,2),result_set.feature_average(mutable_attr, {'group':1})))
    diff = abs(post_up_mean - post_p_mean)
    diff_pre = abs(pre_up_mean - pre_p_mean)
    print("Feature x (mean):")
    print("(UP) Pre :", pre_up_mean, "(+-", pre_up_std, ")")
    print("(P) Pre  :", pre_p_mean, "(+-", pre_p_std, ")")
    print("(UP) Post:", post_up_mean, "(+-", post_up_std, ")")
    print("(P) Post :", post_p_mean, "(+-", post_p_std, ")")

    print("Post    Δ:", round(diff, 2))
    print("Pre     Δ:", round(diff_pre, 2))

    sns.set()
    #plt.figure(name)
    #ax = sns.lineplot(x=mutable_attr, y="incentive",hue='uid',data=(rs._avg_incentive(mutable_attr, 'uid')).reset_index(), legend=False)
    #ax = sns.lineplot(x=mutable_attr, y="incentive",hue='group',data=(rs._avg_incentive(mutable_attr, 'group')).reset_index())
    #plt.show()
    #plt.savefig(name+".png")



privileged_groups = [{'group': 1}]
unprivileged_groups = [{'group': 0}]

means = [40,60]
threshold = 60
for i in range(1000):
    print("Iteration ",i,", threshold:",threshold)
#LogisticLearner(exclude_protected=True)
#ReweighingLogisticLearner(privileged_groups, unprivileged_groups)
#StatisticalParityLogisticLearner(privileged_groups, unprivileged_groups, eps=0.001)
    rs_aff = do_sim(LogisticLearner(exclude_protected=True), means=means,threshold=threshold)
    rs = do_sim(StatisticalParityLogisticLearner(privileged_groups, unprivileged_groups, eps=0.01, exclude_protected=True), means=means,threshold=threshold)
    _, _, post_up_mean, _ = rs.feature_average(mutable_attr, {'group':0})
    _, _, post_p_mean, _ = rs.feature_average(mutable_attr, {'group':1})
    means = [post_up_mean, post_p_mean]
    threshold += 1

    if i % 5 == 0:
        ax = sns.lineplot(x=mutable_attr, y="incentive",hue='group',data=(rs._avg_incentive(mutable_attr, 'group')).reset_index())
        ax = sns.lineplot(x=mutable_attr, y="incentive",hue='group',data=(rs_aff._avg_incentive(mutable_attr, 'group')).reset_index())
        plt.show()

    print_stats(rs, "noaff")
    print("\n")


