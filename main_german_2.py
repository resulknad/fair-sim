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
from mutabledataset import GermanSimDataset
import matplotlib.pyplot as plt
import seaborn as sns

mutable_attrs = ['savings', 'credit_amount', 'month', 'savings', 'investment_as_income_percentage']

def do_sim(learner):
    L = 0.1
    M = 0.3
    cp = lambda x_new, x: (pow(x_new/2.,2.)*(1-M)+abs(x_new-x)*M)*L #lambda x_new, x: x_new/2.+1*abs(x_new-x)/4.
    cn = lambda x_new, x: (pow((1-x_new)/2.,2.)*(1-M)+abs(x_new-x)*M)*L #lambda x_new, x: x_new/2.+1*abs(x_new-x)/4.
    cost_fixed = lambda size: np.abs(np.random.normal(loc=0.5,size=size))

    g = GermanSimDataset(mutable_features=mutable_attrs,
            domains={k: 'auto' for k in mutable_attrs},
                         discrete=mutable_attrs,
                         protected_attribute_names=['age'],
                         cost_fns={'number_of_credits': cn, 'credit_amount': cn, 'month': cn, 'savings': cp, 'investment_as_income_percentage': cp},
                         privileged_classes=[lambda x: x >= 25],
                         features_to_drop=['personal_status', 'sex'])

    privileged_groups = [{'age': 1}]
    unprivileged_groups = [{'age': 0}]

    sim = Simulation(g,
                     RationalAgent,
                     learner,
                     cost_fixed,
                     split=[0.9])

    #g = SimpleDataset(mutable_features=['x'],
    #        domains={'x': 'auto'},
    #        discrete=['x'],
    #        cost_fns={'x': lambda x_new, x: pow(x_new/2.,2.)+3*abs(x_new-x)/4.})


    #sim = Simulation(g,
    #                 RationalAgent,
    #                 learner,
    #                 cost_fixed)

    result_set = sim.start_simulation(runs=3)
    return result_set

def print_stats(result_set, name):
    print(result_set)
    print("StatPar Δ:", round(result_set.stat_parity_diff({'age': 0}, {'age': 1}),2))
    #print("Feature x (mean):")
    #print("(UP) Pre :", pre_up_mean, "(+-", pre_up_std, ")")
    #print("(P) Pre  :", pre_p_mean, "(+-", pre_p_std, ")")
    #print("(UP) Post:", post_up_mean, "(+-", post_up_std, ")")
    #print("(P) Post :", post_p_mean, "(+-", post_p_std, ")")
    pre_up_mean, pre_up_std, post_up_mean, post_up_std = tuple(map(lambda x: round(x,2),result_set.feature_average('credit', {'age':0})))
    pre_p_mean, pre_p_std, post_p_mean, post_p_std = tuple(map(lambda x: round(x,2),result_set.feature_average('credit', {'age':1})))
    print('y (up):',pre_up_mean, post_up_mean)
    print('y (p):', pre_p_mean, post_p_mean)

    diff = 0
    movement = []
    movement_up = []
    for mutable_attr in mutable_attrs:
        pre_up_mean, pre_up_std, post_up_mean, post_up_std = tuple(map(lambda x: round(x,2),result_set.feature_average(mutable_attr, {'age':0})))
        pre_p_mean, pre_p_std, post_p_mean, post_p_std = tuple(map(lambda x: round(x,2),result_set.feature_average(mutable_attr, {'age':1})))
        diff_pre = abs(pre_p_mean-pre_up_mean)
        diff += abs(post_p_mean-post_up_mean)
        movement.append(pre_p_mean-post_p_mean)
        movement_up.append(pre_up_mean-post_up_mean)
    print("Post    Δ:", round(diff, 2))
    print("Pre     Δ:", round(diff_pre,2))
    print("Movement (p,up):", movement, movement_up)

    #sns.set()
    #plt.figure(name)
    #ax = sns.lineplot(x=mutable_attr, y="incentive",hue='age',data=(rs._avg_incentive(mutable_attr, 'age')).reset_index())
    #plt.savefig(name+".png")

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

print("No aff. action:")
rs = do_sim(LogisticLearner())
print_stats(rs, "noaff")
print("\n")

print("Aff. action (DIY Stat. Parity enforcer)")
rs = do_sim(StatisticalParityLogisticLearner(privileged_groups, unprivileged_groups, eps=0.001))
print_stats(rs, "postDIY")

print("Aff. action (Fairlearn learner)")
rs = do_sim(FairLearnLearner(privileged_groups, unprivileged_groups))
print_stats(rs, "in")
print("\n")



print("Aff. action (Reweighing)")
rs = do_sim(ReweighingLogisticLearner(privileged_groups, unprivileged_groups))
print_stats(rs, "pre")
print("\n")

print("Aff. action (Reject Option)")
rs = do_sim(RejectOptionsLogisticLearner(privileged_groups, unprivileged_groups))
print_stats(rs, "post")
print("\n")
