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
from sklearn.preprocessing import MaxAbsScaler
from agent import RationalAgent, RationalAgentOrig
from simulation import Simulation, SimulationResultSet
from learner import LogisticLearner
import plot
import numpy as np
import pandas as pd
from learner import StatisticalParityLogisticLearner, StatisticalParityFlipperLogisticLearner
from learner import FairLearnLearner
from learner import GaussianNBLearner
from learner import RejectOptionsLogisticLearner, MetaFairLearner
from learner import ReweighingLogisticLearner
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import normal
from IPython.display import display, Markdown, Latex
from scipy.special import huber
from utils import _df_selection, count_df

sns.set(rc={'figure.figsize':(20.7,8.27)})
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
all_mutable = mutable_monotone_pos + mutable_monotone_neg + mutable_dontknow + categorical
all_mutable_dedummy = list(set(list(map(lambda s: s.split('=')[0], all_mutable))))
COST_CONST = 12. #len(all_mutable)
c_pos = lambda x_new, x, rank: softmax((rank(x_new)-rank(x)), 0.)/COST_CONST
c_neg = lambda x_new, x, rank: softmax((rank(x)-rank(x_new)), 0.)/COST_CONST
c_cat = lambda x_new, x, rank: np.abs(rank(x_new)-rank(x))/COST_CONST
c_immutable = lambda x_new, x, rank: np.abs(x_new-x)*np.nan_to_num(float('inf'))


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

C_EXECUTE = True

# # Notions of Fairness

# +
# Compare different notions of fairness

data = dataset()
COST_CONST = 8
data = dataset()
learners = [("no constraint", LogisticLearner(exclude_protected=True)),
            ("statistical parity", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group])),
            ("AvOdds", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], metric_name='Average odds difference'))]


if C_EXECUTE:
    # execute
    rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=60)), learners))
    # save
    save(rss, "notions_of_fairness_" + str(COST_CONST))

# +
rss = load("notions_of_fairness_8")

for ft in all_mutable_dedummy:
    for name, rs in rss:
        display(Markdown("#### " + ft + ", " + name))
        plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, [ft], name=name, kind='cdf')

plot.boxplot(rss, unprivileged_group, privileged_group)

# -
# # Statistical Parity Comparison

# +
data = dataset()
COST_CONST = 8
learners = [("baseline", LogisticLearner(exclude_protected=False)),
    ("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group])),
    ("in",FairLearnLearner([privileged_group], [unprivileged_group])),
    ("post",RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))]

if C_EXECUTE:
    # execute
    rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=60)), learners))
    # save
    save(rss, "statpar_comp_cost_" + str(COST_CONST))


# +
rss = load("statpar_comp_cost_8")

for ft in all_mutable_dedummy:
    for name, rs in rss:
        display(Markdown("#### " + ft + ", " + name))
        plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, [ft], name=name, kind='cdf')

plot.boxplot(rss, unprivileged_group, privileged_group)
# -



# # Colorblind vs Colorsighted

# +
data = dataset()
COST_CONST = 8

learners = [("logreg cb", LogisticLearner(exclude_protected=True)),
    ("logreg cs", LogisticLearner(exclude_protected=False)),
    ("nb cb",GaussianNBLearner(exclude_protected=True)),
    ("nb cs",GaussianNBLearner(exclude_protected=False))]

if C_EXECUTE:
    # execute
    rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=60)), learners))
    # save
    save(rss, "colorsigted_blind_" + str(COST_CONST))


# +
rss = load("colorsigted_blind_8")

for ft in all_mutable_dedummy:
    for name, rs in rss:
        display(Markdown("#### " + ft + ", " + name))
        plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, [ft], name=name, kind='cdf')

plot.boxplot(rss, unprivileged_group, privileged_group)
