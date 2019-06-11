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

import pickle
from mutabledataset import GermanSimDataset
from itertools import starmap
from sklearn.preprocessing import MaxAbsScaler
from agent import RationalAgent, RationalAgentOrig
from simulation import Simulation, SimulationResultSet
from learner import LogisticLearner
import plot
import numpy as np
import pandas as pd
from learner import StatisticalParityLogisticLearner, StatisticalParityFlipperLogisticLearner
from learner import FairLearnLearner
from learner import RandomForestLearner, GaussianNBLearner
from learner import RejectOptionsLogisticLearner, MetaFairLearner
from learner import ReweighingLogisticLearner
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import normal
from IPython.display import display, Markdown, Latex
from scipy.special import huber
from plot import _df_selection, count_df

sns.set(rc={'figure.figsize':(20.7,8.27)})
# -

# ## Parameters for simulation

# +

immutable = ['age']

mutable_monotone_neg = ['month', 'credit_amount', 'status', 'investment_as_income_percentage', 'number_of_credits', 'people_liable_for']
mutable_dontknow = ['residence_since']
mutable_monotone_pos = ['savings']

#categorical = ['credit_history=A34', 'purpose=A48','has_checking_account', 'purpose=A41', 'other_debtors=A103', 'purpose=A46', 'purpose=A40', 'credit_history=A31', 'employment=A74', 'credit_history=A30', 'credit_history=A33', 'purpose=A410', 'installment_plans=A143', 'housing=A153', 'property=A121', 'telephone=A192', 'skill_level=A171', 'purpose=A44', 'purpose=A45', 'housing=A152', 'other_debtors=A102', 'employment=A75', 'employment=A71', 'purpose=A43', 'property=A124', 'property=A123', 'housing=A151', 'employment=A72', 'credit_history=A32', 'property=A122', 'telephone=A191', 'installment_plans=A142', 'skill_level=A172', 'purpose=A42', 'employment=A73', 'other_debtors=A101', 'skill_level=A173', 'purpose=A49', 'installment_plans=A141', 'skill_level=A174']


mutable_attr = 'savings'
group_attr = 'age'
priv_classes = [lambda x: x >= 25]

privileged_group = {group_attr: 1}
unprivileged_group = {group_attr: 0}
DefaultAgent = RationalAgent #RationalAgent, RationalAgentOrig (approx b(x) = h(x))

# CURRENT PROBLEM: subsidy doesnt help with stat parity...
# IDEA: if we subsidize heavily, the motivation to move across the boundary is too low
# especially because of the stopping criteria (direction should stay the same as subsidy is constant)

cost_fixed = lambda size: np.array([0] * size) #lambda size: np.abs(np.random.normal(loc=1,scale=0.5,size=size))# np.array([0.] * size) #np.abs(np.random.normal(loc=0,scale=0,size=size)) #np.abs(np.random.normal(loc=1,scale=0.5,size=size))
# TODO: plot cost function for dataset

C = 0.1


# https://en.wikipedia.org/wiki/Smooth_maximum
# differentiable function approximating max
def softmax(x1, x2):
    return np.maximum(x1,x2)
    alpha = 1
    e_x1 = np.exp(alpha*x1)
    e_x2 = np.exp(alpha*x2)
    return (x1*e_x1 + x2*e_x2)/(e_x1+e_x2)

# /len(all_mutable)
all_mutable = mutable_monotone_pos + mutable_monotone_neg + mutable_dontknow# + categorical
COST_CONST = 12. #len(all_mutable)
c_pos = lambda x_new, x, rank: softmax((rank(x_new)-rank(x)), 0.)/COST_CONST
c_neg = lambda x_new, x, rank: softmax((rank(x)-rank(x_new)), 0.)/COST_CONST
c_cat = lambda x_new, x, rank: np.abs(rank(x_new)-rank(x))/COST_CONST
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
                             #**{a: c_cat for a in categorical},
                             **{a: c_immutable for a in immutable}},
                         privileged_classes=priv_classes,
                         features_to_drop=['personal_status', 'sex', 'foreign_worker'])
def do_sim(learner, cost_fixed=cost_fixed, cost_fixed_dep=None, collect_incentive_data=False, no_neighbors=60):
    data = dataset()
    sim = Simulation(data,
                     DefaultAgent,
                     learner,
                     cost_fixed if cost_fixed_dep is None else None,
                     collect_incentive_data=collect_incentive_data,
                     avg_out_incentive=1,
                     no_neighbors=no_neighbors,
                     cost_distribution_dep=cost_fixed_dep,
                     split=[0.9])

    result_set = sim.start_simulation(runs=1)
    return result_set

def save(obj, filename):
    dumpfile = open(filename, 'wb')
    pickle.dump(obj, dumpfile)
    dumpfile.close()

def load(filename):
    rs = {}
    dumpfile = open(filename, 'rb')
    rs = pickle.load(dumpfile)
    dumpfile.close()
    return rs


# -

# ## LogReg feature coefficients

# +
data = dataset()

l = LogisticLearner()
l.fit(data)

display(Markdown("#### LogReg Coeffs."))
#display(pd.DataFrame(columns=['Feature', 'Coefficient LogReg'], data=l.coefs))

dist_plot_attr = 'savings'
data.infer_domain()
fns = data.rank_fns()
if False:
    sample = np.linspace(-1,10,100)
    data_arr = list(map(fns[0][dist_plot_attr], sample))
    data_arr.extend(list(map(fns[0][dist_plot_attr], sample)))
    data_arr = np.array([np.hstack((sample,sample)), data_arr]).transpose()
    df = pd.DataFrame(data=data_arr, columns=['x', 'y'])
    ax = sns.lineplot(x='x', y="y",data=df)
    display(Markdown("### Distribution of " + dist_plot_attr))
    plt.show()


    # -

    # ## Logistic Learner
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
display(Markdown("#### (log reg) distribution pre/post sim for " + mutable_attr))


# -

def plot_ga(rs, index):
    # benefit, cost, incentive_mean graph
    d = rs.results[0].incentives
    #np.argmax(np.array(d[0][0])[:,3] - np.array(d[len(d)-1][0])[:,3])
    savings = list(starmap(lambda i,x: [i, np.mean(x['features'][:,1][index])], zip(range(len(d)), d)))
    benefit = list(starmap(lambda i,x: [i, np.mean(x['benefit'][index])], zip(range(len(d)), d)))
    boost = list(starmap(lambda i,x: [i, x['boost']], zip(range(len(d)), d)))
    incentive_mean = list(starmap(lambda i,x: [i, np.mean(x['benefit'])-np.mean(x['cost'])], zip(range(len(d)), d)))
    cost = list(starmap(lambda i,x: [i, np.mean(x['cost'][index])], zip(range(len(d)), d)))
    df = pd.DataFrame(data=(np.vstack((savings,
                                       benefit,
                                       cost,
                                       #incentive_mean,
                                       boost))),
                      columns=["t", "val"],
                      index=(["month"]*len(d)
                             + ["benefit"]*len(d)
                             + ["cost"] * len(d)
                             #+ ["incentive_mean"] * len(d)
                             + ["boost"] * len(d))).reset_index()
    plt.figure()
    ax = sns.lineplot(x='t', y="val", hue='index',data=df)
    plt.show()


# +

data = dataset()
#rs = do_sim(FairLearnLearner([privileged_group], [unprivileged_group]), collect_incentive_data=True)
#rs = do_sim(LogisticLearner([privileged_group], [unprivileged_group]), collect_incentive_data=True)
COST_CONST = 8
#rs_log = do_sim(LogisticLearner(exclude_protected=False), no_neighbors=1, collect_incentive_data=True)

rs_ro = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]), no_neighbors=60, collect_incentive_data=True)


#COST_CONST = 3
#rs3 = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]), no_neighbors=100, collect_incentive_data=True)

#save([("3", rs16), ("16", rs16)], "statpar_invest")

#plot_all_mutable_features(rs)


#data = dataset()
#rs = do_sim(LogisticLearner(exclude_protected=True), collect_incentive_data=False)
#plot_all_mutable_features(rs)


# +
plot_ga(rs_ro, 11)
plot.boxplot([("ro", rs_ro)], unprivileged_group, privileged_group)

#index = 5
#rss = load("statpar_invest")
#display(Markdown("# cost 3"))
#plot_ga(rss[0][1], index)

#display(Markdown("# cost 16"))
#plot_ga(rss[1][1], index)
# -

plot.boxplot([("a", rs)], unprivileged_group, privileged_group)

# ## Comparison of different predictive methods

if False:
    y_attr = 'credit_h' #mutable_attr # credit_h for prediction, credit for updated ground truth
    def extract_avg_ft(rs, ft, name):
        pre_p_mean, _, post_p_mean, _ = rs.feature_average(y_attr, {})
        return [[name, 'pre', pre_p_mean], [name, 'post', post_p_mean]]


    # +
    plot_data = []
    rs = do_sim(LogisticLearner(exclude_protected=True))
    plot_data = extract_avg_ft(rs, y_attr, "LogReg")
    display(Markdown("#### Feature averages per group LogReg protected excluded"))
    display(rs.feature_table([unprivileged_group, privileged_group]))
    #rs = do_sim(RandomForestLearner())
    #plot_data += extract_avg_ft(rs, y_attr, "RandomForest")

    rs = do_sim(GaussianNBLearner())
    display(Markdown("#### Feature averages per group mutlinomial nb protexcl"))
    display(rs.feature_table([unprivileged_group, privileged_group]))
    plot_data += extract_avg_ft(rs, y_attr, "GaussianNB")

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])
    plt.figure()
    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.show()
    # -

    # ## Color-sighted vs. color-blind

    # +

    metric_name = "statpar" #gtdiff, mutablediff



    plot_data = []
    rs = do_sim(LogisticLearner(exclude_protected=True))
    plot_data.append(["LogReg", False, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(LogisticLearner())
    plot_data.append(["LogReg", True, extract_metric(rs, metric_name=metric_name)])

    #rs = do_sim(RandomForestLearner())
    #plot_data.append(["RandomForest", True, extract_metric(rs, metric_name=metric_name)])

    #rs = do_sim(RandomForestLearner(exclude_protected=True))
    #plot_data.append(["RandomForest", False, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(GaussianNBLearner())
    plot_data.append(["GaussianNB", True, extract_metric(rs, metric_name=metric_name)])

    rs = do_sim(GaussianNBLearner(exclude_protected=True))
    plot_data.append(["GaussianNB", False, extract_metric(rs, metric_name=metric_name)])

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "protectedincluded", metric_name])

    plt.figure()
    sns.catplot(x="name", y=metric_name, hue="protectedincluded", kind="bar",
                data=plot_data_df)
    plt.savefig("sighted_comp.png")


# # Iterations

# +
# Compare different notions of fairness

data = dataset()
learners = [("baseline_wp", LogisticLearner(exclude_protected=False)),
    ("baseline_wop", LogisticLearner(exclude_protected=True)),
    ("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group]))]
rss = []
for i in range(10):
    COST_CONST = 10+i
    rs =  do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]), no_neighbors=60)
    #plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, all_mutable, name=name)
    rss.append((str(COST_CONST), rs))
# execute
#rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=60)), learners))
plot.boxplot(rss, unprivileged_group, privileged_group)

# save
save(rss, "cost_const_ro_statpar")

# +
# load
rss = load("iterations")

#for name, rs in rss:
#    display(Markdown("## " + name))

rs = rss[len(rss)-1][1]
plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, all_mutable, name=name)
sns.set(rc={'figure.figsize':(20.7,8.27)})
plot.boxplot(rss, unprivileged_group, privileged_group)
# -

# # Notions of Fairness

# +
# Compare different notions of fairness

data = dataset()
learners = [("no constraint", LogisticLearner(exclude_protected=True)),
            ("statistical parity", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group])),
            ("AvOdds", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], metric_name='Average odds difference'))]

# execute
rss = list(map(lambda x: (x[0],do_sim(x[1])), learners))

# save
save(rss, "notions_of_fairness_2")
# + {}
# load
rss = load("notions_of_fairness_2")

for name, rs in rss:
    display(Markdown("## " + name))
    #plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, all_mutable, name=name)
    
plot.boxplot(rss, unprivileged_group, privileged_group)

# -




# ## Statistical parity comparison
# with protected attributes included, otherwise it's boring...
#
# gtdiff: difference of averages (of priv and unpriv) of the new ground truth (assigned by KNN with custom cost function as distance)

# +
data = dataset()

COST_CONST = 8.
learners = [("baseline", LogisticLearner(exclude_protected=False)),
    ("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group])),
    ("in",FairLearnLearner([privileged_group], [unprivileged_group])),
    ("post",RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))]

# execute
rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=60)), learners))

# save
save(rss, "statpar_comp_cost2_ra")

# -



# +
# load
rss = load("statpar_comp_cost2_ra")
df = rss[0][1].results[0].df
print(np.mean(df.loc[df['age'] == 1,'credit_h_pr']))
for ft in all_mutable:
    for name, rs in rss:
        #display(Markdown("## " + name))
        #if name == 'post':
        #print(rs.feature_table([unprivileged_group, privileged_group]))
        plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, [ft], name=name, kind='cdf')

plot.boxplot(rss, unprivileged_group, privileged_group)

# +
# load
rss = load("statpar_comp_cost12_2")

for name, rs in rss:
    display(Markdown("## " + name))
    #if name == 'post':
    plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, ['month'], name=name, kind='cdf')
    
plot.boxplot(rss, unprivileged_group, privileged_group)
# -

# ## Time of Intervention
#
# - Pre: assigning a fixed cost of -subsidy to unprivileged group
# - Post: setting the bounds for acceptable statistical parity difference to 1-subsidy (high subsidy means stat parity diff must be 0)

data = dataset()
data.feature_names.index('age')



# +
data = dataset()

metric_name = 'statpar' # statpar, gtdiff
subsidies = np.linspace(0,1,4)

learner = LogisticLearner(exclude_protected=False)

# creates a instance dependant cost lambda fn
# which reduces cost for unprivileged group by subsidy
def cost_fixed_dependant(subsidy):
    group_attr_ind = data._ft_index(group_attr)
    privileged_val = privileged_group[group_attr]

    return lambda x: cost_fixed(1)[0] if x[group_attr_ind] == privileged_val else cost_fixed(1)[0] - subsidy*1.


result_sets = list(map(lambda s: do_sim(learner, cost_fixed_dep=cost_fixed_dependant(s), collect_incentive_data=True), subsidies))

#for rs in result_sets:
    #display(rs.feature_table([unprivileged_group, privileged_group]))
    #print(rs)

result_sets += list(map(lambda s: do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], abs_bound=s)), reversed(subsidies)))


y_values = list(map(lambda rs: extract_metric(rs, metric_name), result_sets))
index = ['pre'] * len(subsidies) + ['post'] * len(subsidies)

x = np.array([np.hstack((subsidies,subsidies)), y_values]).transpose()
#x = np.array([subsidies, y_values]).transpose()

# subsidy graph
plot_data_df = pd.DataFrame(x, columns=['subsidy', metric_name],index=index).reset_index()
plt.figure()
ax = sns.lineplot(x='subsidy',hue='index', y=metric_name, data=plot_data_df)

# benefit, cost, incentive_mean graph
#for rs in result_sets:
#    d = rs.results[0].incentives
#    index = 147 #np.argmax(np.array(d[0][0])[:,3] - np.array(d[len(d)-1][0])[:,3])
#    savings = list(starmap(lambda i,x: [i, np.mean(x[0][:,3])], zip(range(len(d)), d)))
#    benefit = list(starmap(lambda i,x: [i, np.mean(x[1])], zip(range(len(d)), d)))
#    incentive_mean = list(starmap(lambda i,x: [i, np.mean(x[1])-np.mean(x[2])], zip(range(len(d)), d)))
#    cost = list(starmap(lambda i,x: [i, np.mean(x[2])], zip(range(len(d)), d)))
#    df = pd.DataFrame(data=(np.vstack((savings,benefit,cost,incentive_mean))), columns=["t", "val"], index=(["savings"]*len(d) + ["benefit"]*len(d) + ["cost"] * len(d)+ ["incentive_mean"] * len(d))).reset_index()
#    plt.figure()
#    ax = sns.lineplot(x='t', y="val", hue='index',data=df)


plt.show()
#for rs in result_sets:
#    ax = sns.lineplot(x=mutable_attr, y="incentive",hue=group_attr,data=(rs.                     _avg_incentive(mutable_attr, group_attr)).reset_index())
#    plt.show()
# -


