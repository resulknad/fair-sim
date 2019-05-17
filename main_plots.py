# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

from mutabledataset import GermanSimDataset
from itertools import starmap
from sklearn.preprocessing import MaxAbsScaler
from agent import RationalAgent, RationalAgentOrig
from simulation import Simulation
from learner import LogisticLearner
import plot
import numpy as np
import pandas as pd
from learner import StatisticalParityLogisticLearner, StatisticalParityFlipperLogisticLearner
from learner import FairLearnLearner
from learner import RandomForestLearner, MultinomialNBLearner
from learner import RejectOptionsLogisticLearner
from learner import ReweighingLogisticLearner
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import normal
from IPython.display import display, Markdown, Latex
from scipy.special import huber

sns.set()
# -

# ## Parameters for simulation

# +

immutable = ['age']

mutable_monotone_neg = ['month', 'credit_amount', 'status', 'investment_as_income_percentage', 'number_of_credits', 'people_liable_for']
mutable_dontknow = ['residence_since']
mutable_monotone_pos = ['savings']

categorical = ['credit_history=A34', 'purpose=A48','has_checking_account', 'purpose=A41', 'other_debtors=A103', 'purpose=A46', 'purpose=A40', 'credit_history=A31', 'employment=A74', 'credit_history=A30', 'credit_history=A33', 'purpose=A410', 'installment_plans=A143', 'housing=A153', 'property=A121', 'telephone=A192', 'skill_level=A171', 'purpose=A44', 'purpose=A45', 'housing=A152', 'other_debtors=A102', 'employment=A75', 'employment=A71', 'purpose=A43', 'property=A124', 'property=A123', 'housing=A151', 'employment=A72', 'credit_history=A32', 'property=A122', 'telephone=A191', 'installment_plans=A142', 'skill_level=A172', 'purpose=A42', 'employment=A73', 'other_debtors=A101', 'skill_level=A173', 'purpose=A49', 'installment_plans=A141', 'skill_level=A174']


mutable_attr = 'savings'
group_attr = 'age'
priv_classes = [lambda x: x >= 25]

privileged_group = {group_attr: 1}
unprivileged_group = {group_attr: 0}
DefaultAgent = RationalAgent #RationalAgent, RationalAgentOrig (approx b(x) = h(x))

# CURRENT PROBLEM: subsidy doesnt help with stat parity...
# IDEA: if we subsidize heavily, the motivation to move across the boundary is too low
# especially because of the stopping criteria (direction should stay the same as subsidy is constant)

cost_fixed = lambda size: np.abs(np.random.normal(loc=1,scale=0.5,size=size))
# TODO: plot cost function for dataset

C = 0.1


# https://en.wikipedia.org/wiki/Smooth_maximum
# differentiable function approximating max
def softmax(x1, x2):
    return np.maximum(x1,x2)
    alpha = 8
    e_x1 = np.exp(alpha*x1)
    e_x2 = np.exp(alpha*x2)
    return (x1*e_x1 + x2*e_x2)/(e_x1+e_x2)


all_mutable = mutable_monotone_pos + mutable_monotone_neg + mutable_dontknow + categorical
c_pos = lambda x_new, x, rank: softmax((rank(x_new)-rank(x))/len(all_mutable), 0.)
c_neg = lambda x_new, x, rank: softmax((rank(x)-rank(x_new))/len(all_mutable), 0.)
c_cat = lambda x_new, x, rank: (C/len(all_mutable))
c_immutable = lambda x_new, x, rank: np.abs(x_new-x)*np.nan_to_num(float('inf'))


print(len(all_mutable))



# -

# ## Common simulation code

# +

def dataset():
    return GermanSimDataset(mutable_features=all_mutable,
            domains={k: 'auto' for k in all_mutable},
                         discrete=all_mutable,
                         protected_attribute_names=[group_attr],
                         cost_fns={ **{a: c_pos for a in mutable_monotone_pos},
                             **{a: c_neg for a in mutable_monotone_neg},
                             **{a: c_cat for a in categorical},
                             **{a: c_immutable for a in immutable}},
                         privileged_classes=priv_classes,
                         features_to_drop=['personal_status', 'sex', 'foreign_worker'])
def do_sim(learner, cost_fixed=cost_fixed, cost_fixed_dep=None, collect_incentive_data=False):
    data = dataset()
    sim = Simulation(data,
                     DefaultAgent,
                     learner,
                     cost_fixed if cost_fixed_dep is None else None,
                     collect_incentive_data=collect_incentive_data,
                     avg_out_incentive=1,
                     no_neighbors=51,
                     cost_distribution_dep=cost_fixed_dep,
                     split=[0.9])

    result_set = sim.start_simulation(runs=1)
    return result_set


    # -

    # ## LogReg feature coefficients

    # +

if False:
    data = dataset()
    l = LogisticLearner()
    l.fit(data)

    display(Markdown("#### LogReg Coeffs."))
    display(pd.DataFrame(columns=['Feature', 'Coefficient LogReg'], data=l.coefs))

    dist_plot_attr = 'savings'
    data.infer_domain()
    fns = data.rank_fns()

    sample = np.linspace(-1,10,100)
    data_arr = list(map(fns[1][dist_plot_attr], sample))
    data_arr.extend(list(map(fns[1][dist_plot_attr], sample)))
    data_arr = np.array([np.hstack((sample,sample)), data_arr]).transpose()
    df = pd.DataFrame(data=data_arr, columns=['x', 'y'])
    ax = sns.lineplot(x='x', y="y",data=df)
    display(Markdown("### Distribution of " + dist_plot_attr))
    plt.show()

    # -

    # ## Cost function?

    rs = do_sim(LogisticLearner(exclude_protected=True), collect_incentive_data=False)
    display(Markdown("#### Feature averages per group LogReg protected excluded"))
    display(rs.feature_table([unprivileged_group, privileged_group]))
    display(Markdown("#### Equilibrium related results"))
    display(str(rs))
    #for grp_avg in (rs.results[0].df_new.groupby([group_attr]).mean().reset_index()):
    #    samples = np.linspace(min(data.domains[mutable_attr]), max(data.domains[mutable_attr]),50)
    #    incentive =

    # +
    #ax = sns.lineplot(x=mutable_attr, y="incentive",hue=group_attr,data=(rs.
    #    _avg_incentive(mutable_attr, group_attr)).reset_index())
    #plt.show()

    # +
    display(Markdown("#### distribution pre/post sim for " + mutable_attr))

    def merge_dfs(col, colval1, colval2, df1, df2):
        df1[col] = pd.Series([colval1] * len(df1.index), df1.index)
        df2[col] = pd.Series([colval2] * len(df2.index), df2.index)
        return pd.concat((df1, df2))
    merged =merge_dfs('time', 'pre', 'post', rs.results[0].df, rs.results[0].df_new)
    sns.catplot(x=mutable_attr, hue="time", kind="count",
                data=merged)
    plt.show()
    # -

    # ## Comparison of different predictive methods

    # +
    y_attr = 'credit' #mutable_attr # credit_h for prediction, credit for updated ground truth
    def extract_avg_ft(rs, ft, name):
        pre_p_mean, _, post_p_mean, _ = rs.feature_average(y_attr, {})
        return [[name, 'pre', pre_p_mean], [name, 'post', post_p_mean]]
    plot_data = []

    rs = do_sim(LogisticLearner(exclude_protected=True))
    plot_data = extract_avg_ft(rs, y_attr, "LogReg")

    rs = do_sim(RandomForestLearner())
    plot_data += extract_avg_ft(rs, y_attr, "RandomForest")

    rs = do_sim(MultinomialNBLearner())
    plot_data += extract_avg_ft(rs, y_attr, "MultinomialNB")

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])
    plt.figure()
    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.show()


    # -

    # ## Color-sighted vs. color-blind

    # +

metric_name = "statpar" #gtdiff, mutablediff

def extract_metric(rs, metric_name="statpar", time='post'):
    if metric_name == "statpar":
        ret = rs.stat_parity_diff(unprivileged_group, privileged_group, time=time )
        return ret
    elif metric_name == 'gtdiff':
        gt_label = data.label_names[0]
        pre_p_mean, _, post_p_mean, _ = rs.feature_average(gt_label, privileged_group)
        pre_up_mean, _, post_up_mean, _ = rs.feature_average(gt_label, unprivileged_group)
        return abs(post_p_mean - post_up_mean) if time == 'post' else abs(pre_p_mean - pre_up_mean)
    elif metric_name == 'mutablediff':
        pre_p_mean, _, post_p_mean, _ = rs.feature_average(mutable_attr, privileged_group)
        pre_up_mean, _, post_up_mean, _ = rs.feature_average(mutable_attr, unprivileged_group)
        return abs(post_p_mean - post_up_mean) if time == 'post' else abs(pre_p_mean - pre_up_mean)

    return None


if False:
    plot_data = []
    rs = do_sim(LogisticLearner(exclude_protected=True))
    plot_data.append(["LogReg", False, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(LogisticLearner())
    plot_data.append(["LogReg", True, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(RandomForestLearner())
    plot_data.append(["RandomForest", True, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(RandomForestLearner(exclude_protected=True))
    plot_data.append(["RandomForest", False, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(MultinomialNBLearner())
    plot_data.append(["MultinomialNB", True, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(MultinomialNBLearner(exclude_protected=True))
    plot_data.append(["MultinomialNB", False, extract_metric(rs, metric_name=metric_name)])

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "protectedincluded", metric_name])

    plt.figure()
    sns.catplot(x="name", y=metric_name, hue="protectedincluded", kind="bar",
                data=plot_data_df)
    plt.savefig("sighted_comp.png")

    # -

    # # Notions of Fairness

    # +

    # Compare different notions of fairness
    y_attr = 'credit'

    rs = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))
    plot_data = extract_avg_ft(rs, mutable_attr, "RO_StatPar")

    rs = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], metric_name='Equal opportunity difference'))
    plot_data += extract_avg_ft(rs, mutable_attr, "RO_EqOpp")

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])


    plt.figure()
    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.show()

    # -

    # ## Statistical parity comparison

    # +

    plot_data = []
    learners = [("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group])),
        ("in",FairLearnLearner([privileged_group], [unprivileged_group])),
        ("post",RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))]

    y_attr = mutable_attr
    y_attr = 'credit'

    for name, l in learners:
        rs = do_sim(l)
        plot_data += extract_avg_ft(rs, mutable_attr, name)

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])

    plt.figure()
    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.show()

# split for groups
# 1. section 2.1 for cost function
# 2. cost function as distance, only within group (immutable)
# -

# ## Time of Intervention

# +
data = dataset()

metric_name = 'statpar' # statpar, gtdiff
subsidies = np.linspace(0,1,2)

learner = LogisticLearner(exclude_protected=False)

# creates a instance dependant cost lambda fn
# which reduces cost for unprivileged group by subsidy
def cost_fixed_dependant(subsidy):
    group_attr_ind = data._ft_index(group_attr)
    privileged_val = privileged_group[group_attr]

    return lambda x: cost_fixed(1)[0] if x[group_attr_ind] == privileged_val else cost_fixed(1)[0] - subsidy*1.


result_sets = list(map(lambda s: do_sim(learner, cost_fixed_dep=cost_fixed_dependant(s), collect_incentive_data=True), subsidies))

for rs in result_sets:
    display(rs.feature_table([unprivileged_group, privileged_group]))
    print(rs)
#result_sets += list(map(lambda s: do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], abs_bound=s)), reversed(subsidies)))


y_values = list(map(lambda rs: extract_metric(rs, metric_name), result_sets))
index = ['pre'] * len(subsidies) #+ ['post'] * len(subsidies)
#x = np.array([np.hstack((subsidies,subsidies)), y_values]).transpose()
x = np.array([subsidies, y_values]).transpose()

plot_data_df = pd.DataFrame(x, columns=['subsidy', metric_name],index=index).reset_index()
print(plot_data_df)
plt.figure()
ax = sns.lineplot(x='subsidy',hue='index', y=metric_name, data=plot_data_df)

for rs in result_sets:
    d = rs.results[0].incentives
    index = np.argmax(np.array(d[0][0])[:,3] - np.array(d[len(d)-1][0])[:,3])
    savings = list(starmap(lambda i,x: [i, np.mean(x[0][index][3])], zip(range(len(d)), d)))
    benefit = list(starmap(lambda i,x: [i, np.mean(x[1][index])], zip(range(len(d)), d)))
    incentive_mean = list(starmap(lambda i,x: [i, np.mean(x[1])-np.mean(x[2])], zip(range(len(d)), d)))
    cost = list(starmap(lambda i,x: [i, np.mean(x[2][index])], zip(range(len(d)), d)))
    df = pd.DataFrame(data=(np.vstack((savings,benefit,cost,incentive_mean))), columns=["t", "val"], index=(["savings"]*len(d) + ["benefit"]*len(d) + ["cost"] * len(d)+ ["incentive_mean"] * len(d))).reset_index()
    print(df)
    plt.figure()
    ax = sns.lineplot(x='t', y="val", hue='index',data=df)


plt.show()
#for rs in result_sets:
#    ax = sns.lineplot(x=mutable_attr, y="incentive",hue=group_attr,data=(rs.                     _avg_incentive(mutable_attr, group_attr)).reset_index())
#    plt.show()
# -


