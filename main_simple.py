# next steps:
# plot incentives (avg per group)
# do grid search, plot outcomes in terms of relevant parameters
# do jupyter story

from mutabledataset import SimpleDataset
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner
import plot
import numpy as np
from learner import StatisticalParityLogisticLearner
from learner import PrejudiceRemoverLearner
from learner import FairLearnLearner
from learner import RejectOptionsLogisticLearner
from learner import ReweighingLogisticLearner
import matplotlib.pyplot as plt
import seaborn as sns

def do_sim(learner):
    cost_lambda = lambda x_new, x: 0#lambda x_new, x: x_new/2.+1*abs(x_new-x)/4.
    cost_fixed = lambda size: np.abs(np.random.normal(loc=0.5,size=size))

    g = SimpleDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={'x': lambda x_new, x: pow(x_new/2.,2.)+3*abs(x_new-x)/4.})


    sim = Simulation(g,
                     RationalAgent,
                     learner,
                     cost_fixed)

    result_set = sim.start_simulation(runs=2)
    return result_set

def print_stats(result_set, name):
    pre_up_mean, pre_up_std, post_up_mean, post_up_std = tuple(map(lambda x: round(x,2),result_set.feature_average('x', {'group':0})))
    pre_p_mean, pre_p_std, post_p_mean, post_p_std = tuple(map(lambda x: round(x,2),result_set.feature_average('x', {'group':1})))


    print(result_set)
    print("StatPar Δ:", round(result_set.stat_parity_diff({'group': 0}, {'group': 1}),2))
    print("Feature x (mean):")
    print("(UP) Pre :", pre_up_mean, "(+-", pre_up_std, ")")
    print("(P) Pre  :", pre_p_mean, "(+-", pre_p_std, ")")
    print("(UP) Post:", post_up_mean, "(+-", post_up_std, ")")
    print("(P) Post :", post_p_mean, "(+-", post_p_std, ")")
    print("Post    Δ:", round(abs(post_p_mean-post_up_mean), 2))

    sns.set()
    plt.figure(name)
    ax = sns.lineplot(x="x", y="incentive",hue='group',data=(rs._avg_incentive('x', 'group')).reset_index())
    plt.savefig(name+".png")



privileged_groups = [{'group': 1}]
unprivileged_groups = [{'group': 0}]




print("Aff. action (Fairlearn learner)")
rs = do_sim(FairLearnLearner(privileged_groups, unprivileged_groups))
print_stats(rs, "in")
print("\n")

print("No aff. action:")
rs = do_sim(LogisticLearner())
print_stats(rs, "noaff")
print("\n")

print("Aff. action (Reweighing)")
rs = do_sim(ReweighingLogisticLearner(privileged_groups, unprivileged_groups))
print_stats(rs, "pre")
print("\n")

print("Aff. action (Reject Option)")
rs = do_sim(RejectOptionsLogisticLearner(privileged_groups, unprivileged_groups))
print_stats(rs, "post")
print("\n")

print("Aff. action (DIY Stat. Parity enforcer)")
rs = do_sim(StatisticalParityLogisticLearner(privileged_groups, unprivileged_groups, eps=0.001))
print_stats(rs, "postDIY")


