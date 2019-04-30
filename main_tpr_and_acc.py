
import nest_asyncio
nest_asyncio.apply()

from mutabledataset import SimpleDataset
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner
import plot
import numpy as np
from learner import RejectOptionsLogisticLearner
from learner import ReweighingLogisticLearner
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import seaborn as sns

mutable_attr = 'x'
def do_sim(learner):
    cost_lambda = lambda x_new, x:  1*abs(x_new-x)/4. + 3*pow(x_new, 2.)/4.
    cost_fixed = lambda size: np.abs(np.random.normal(loc=0.5,scale=0.5,size=size))


    g = SimpleDataset(mutable_features=['x'],
            domains={'x': 'auto'},
            discrete=['x'],
            cost_fns={'x': cost_lambda})

    sim = Simulation(g,
                     RationalAgent,
                     learner,
                     cost_fixed,
                     collect_incentive_data=True, avg_out_incentive=1, no_neighbors=11)

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
    #print("(UP) Pre :", pre_up_mean, "(+-", pre_up_std, ")")
    #print("(P) Pre  :", pre_p_mean, "(+-", pre_p_std, ")")
    print("(UP) Post:", post_up_mean, "(+-", post_up_std, ")")
    print("(P) Post :", post_p_mean, "(+-", post_p_std, ")")

    print("Post    Δ:", round(diff, 2))
    print("Pre     Δ:", round(diff_pre, 2))

    pre_tpr_up, _ = result_set.tpr({'group': 0}, time='pre')
    pre_tpr_p, _ = result_set.tpr({'group': 1}, time='pre')
    print("(UP) pre TPR:",pre_tpr_up)
    print("(P) pre TPR:",pre_tpr_p)

    post_tpr_up, _ = result_set.tpr({'group': 0})
    post_tpr_p, _ = result_set.tpr({'group': 1})
    print("(UP) post TPR:",post_tpr_up)
    print("(P) post TPR:",post_tpr_p)

    sns.set()
    plt.figure(name)
    #ax = sns.lineplot(x=mutable_attr, y="incentive",hue='uid',data=(rs._avg_incentive(mutable_attr, 'uid')).reset_index(), legend=False)
    ax = sns.lineplot(x=mutable_attr, y="incentive",hue='group',data=(rs._avg_incentive(mutable_attr, 'group')).reset_index())
    plt.savefig(name+".png")



privileged_groups = [{'group': 1}]
unprivileged_groups = [{'group': 0}]



print("Reject Options TPR")
rs = do_sim(RejectOptionsLogisticLearner(privileged_groups, unprivileged_groups, metric_name='Equal opportunity difference'))
print_stats(rs, "postRejOpTPR")
print("\n")

print("No aff. action")
rs = do_sim(LogisticLearner(exclude_protected=True))
print_stats(rs, "noaff")
print("\n")

