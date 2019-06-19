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
from learner import RejectOptionsLogisticLearner, MetaFairLearner, CalibratedLogisticLearner
from learner import ReweighingLogisticLearner
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import normal
from IPython.display import display, Markdown, Latex
from scipy.special import huber
from utils import _df_selection, count_df

#sns.set(rc={'figure.figsize':(20.7,8.27)})
# -

# ## Parameters for simulation

# +

immutable = ['age']

mutable_monotone_neg = ['month', 'credit_amount', 'status', 'investment_as_income_percentage', 'number_of_credits', 'people_liable_for']
mutable_dontknow = ['residence_since']
mutable_monotone_pos = ['savings']

categorical = [#'credit_history=A30',
               #'credit_history=A31',
               #'credit_history=A32',
               #'credit_history=A33',
               #'credit_history=A34',
               #'employment=A71',
               #'employment=A72',
               #'employment=A73',
               #'employment=A74',
               #'employment=A75',
               #'has_checking_account',
               #'housing=A151',
               #'housing=A152',
               #'housing=A153',
               #'installment_plans=A141',
               #'installment_plans=A142',
               #'installment_plans=A143',
               #'other_debtors=A101',
               #'other_debtors=A102',
               #'other_debtors=A103',
               #'property=A121',
               #'property=A122',
               #'property=A123',
               #'property=A124',
               'purpose=A40',
               'purpose=A41',
               'purpose=A410',
               'purpose=A42',
               'purpose=A43',
               'purpose=A44',
               'purpose=A45',
               'purpose=A46',
               'purpose=A48',
               'purpose=A49']
               #'skill_level=A171',
               #'skill_level=A172',
               #'skill_level=A173',
               #'skill_level=A174',
               #'telephone=A191',
               #'telephone=A192']



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

C = 0.25


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
c_cat = lambda x_new, x, rank: softmax(x-x_new, x_new-x) * C/COST_CONST
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

C_EXECUTE = False
data = dataset()


# +
# stability of calibration...
def filter_coefs(l):
    return list(filter(lambda x: x[0].startswith('purpose'), l))
COST_CONST = 8
data = dataset()
rss = []
if C_EXECUTE:
    for i in range(1):
        ll = LogisticLearner(exclude_protected=True)
        ll.fit(dataset())
        display(filter_coefs(ll.coefs))
        cl = CalibratedLogisticLearner([privileged_group], [unprivileged_group])
        cl.fit(dataset())
        #display(list(map(filter_coefs, cl.coefs)))
        rss.append(('logreg', do_sim(LogisticLearner(exclude_protected=True), no_neighbors=1, collect_incentive_data=True)))
        rss.append(('cll', do_sim(CalibratedLogisticLearner([privileged_group], [unprivileged_group]), no_neighbors=1, collect_incentive_data=True)))

# save
#save(rss, "calib_expl_" + str(COST_CONST))



# +
index = 1
features = ['purpose=A40', 'purpose=A41']
rss = load("calib_expl_8")

plot.plot_distribution(dataset(), 'month')

plot.plot_ga(rss[0][1], index, features=features)
plot.plot_ga(rss[1][1], index, features=features)
plot.boxplot(rss, unprivileged_group, privileged_group, name="calib_expl_8")

display(rss[0][1].feature_table([unprivileged_group, privileged_group]))
display(rss[1][1].feature_table([unprivileged_group, privileged_group]))

for ft in all_mutable_dedummy:
    for name, rs in rss:
        
        display(Markdown("#### " + ft + ", " + name))
        plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, [ft], name=name, kind='cdf')

plot.boxplot(rss, unprivileged_group, privileged_group)

# -

# # Notions of Fairness

# +
# Compare different notions of fairness

data = dataset()
COST_CONST = 2
data = dataset()
learners = [("no constraint", LogisticLearner(exclude_protected=True)),
            ("calibration", CalibratedLogisticLearner([privileged_group], [unprivileged_group])),
            ("statistical parity", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group])),
            ("average odds", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], metric_name='Average odds difference'))]


if C_EXECUTE:
    # execute
    rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=130)), learners))
    # save
    save(rss, "notions_of_fairness_" + str(COST_CONST))

# +
filename = "notions_of_fairness_8"
rss = load(filename)

for ft in all_mutable_dedummy:
    for name, rs in rss:
        display(Markdown("#### " + ft + ", " + name))
        plot.plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset, [ft], name=filename+'_'+name, kind='cdf')

plot.boxplot(rss, unprivileged_group, privileged_group, name=filename)

# -
# # Statistical Parity Comparison

data = dataset()
COST_CONST = 2
learners = [("baseline", LogisticLearner(exclude_protected=True)),
    ("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group])),
    ("in",FairLearnLearner([privileged_group], [unprivileged_group])),
    ("post",RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))]
if C_EXECUTE:
    # execute
    rss = list(map(lambda x: (x[0],do_sim(x[1], no_neighbors=130)), learners))
    # save
    save(rss, "statpar_comp_cost_" + str(COST_CONST))


indx = 27
print(rss[0][1].results[0].df['age'][indx])
plot.plot_ga(rss[0][1], indx)
print(rss[0][1].results[0].df_new['credit_h_pr'][indx])


dict(zip(dataset().feature_names, rss[0][1].results[0].incentives[61]['features'][indx]-rss[0][1].results[0].incentives[59]['features'][indx]))

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
# -


