import nest_asyncio
nest_asyncio.apply()

from mutabledataset import GermanSimDataset
from sklearn.preprocessing import MaxAbsScaler
from agent import RationalAgent
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

sns.set()

mutable_attr = 'savings'
group_attr = 'age'
priv_classes = [lambda x: x >= 25]

privileged_group = {group_attr: 1}
unprivileged_group = {group_attr: 0}

cost_fixed = lambda size: np.abs(np.random.normal(loc=0,scale=0.5,size=size))
# TODO: plot cost function for dataset

M = 0.2
L = 0.9
cp = lambda x_new, x: (pow(x_new/2.,2.)*(1-M)+abs(x_new-x)*M)*L

data = GermanSimDataset(mutable_features=[mutable_attr],
        domains={mutable_attr: 'auto'},
                     discrete=[mutable_attr],
                     protected_attribute_names=[group_attr],
                     cost_fns={mutable_attr: cp},
                     privileged_classes=priv_classes,
                     features_to_drop=['personal_status', 'sex', 'foreign_worker'])

def do_sim(learner, cost_fixed=cost_fixed, cost_fixed_dep=None):
    sim = Simulation(data,
                     RationalAgent,
                     learner,
                     cost_fixed if cost_fixed_dep is None else None,
                     collect_incentive_data=True,
                     avg_out_incentive=1,
                     no_neighbors=51,
                     cost_distribution_dep=cost_fixed_dep,
                     split=[0.9])

    result_set = sim.start_simulation(runs=1)
    return result_set
    # compare different predictive methods

y_attr = 'credit' #mutable_attr # credit_h for prediction, credit for updated ground truth

def extract_avg_ft(rs, ft, name):
    pre_p_mean, _, post_p_mean, _ = rs.feature_average(y_attr, {})
    return [[name, 'pre', pre_p_mean], [name, 'post', post_p_mean]]
plot_data = []
if False:
    rs = do_sim(LogisticLearner(exclude_protected=True))
    plot_data += extract_avg_ft(rs, y_attr, "LogReg")

    rs = do_sim(RandomForestLearner())
    plot_data += extract_avg_ft(rs, y_attr, "RandomForest")

    rs = do_sim(MultinomialNBLearner())
    plot_data += extract_avg_ft(rs, y_attr, "MultinomialNB")

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])
    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.savefig("methods_comparision.png")
    plt.show()

# compare color-sighted vs. color-blind
metric_name = "statpar"

def extract_metric(rs, metric_name="statpar"):
    if metric_name == "statpar":
        ret = rs.stat_parity_diff(unprivileged_group, privileged_group)
        return ret
    elif metric_name == 'gtdiff':
        gt_label = data.label_names[0]
        _, _, post_p_mean, _ = rs.feature_average(gt_label, privileged_group)
        _, _, post_up_mean, _ = rs.feature_average(gt_label, unprivileged_group)
        return abs(post_p_mean - post_up_mean)
    elif metric_name == 'mutablediff':
        _, _, post_p_mean, _ = rs.feature_average(mutable_attr, privileged_group)
        _, _, post_up_mean, _ = rs.feature_average(mutable_attr, unprivileged_group)
        return abs(post_p_mean - post_up_mean)

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

    sns.catplot(x="name", y=metric_name, hue="protectedincluded", kind="bar",
                data=plot_data_df)
    plt.show()
    plt.savefig("sighted_comp.png")

# Compare different notions of fairness
y_attr = mutable_attr
if False:
    rs = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))
    plot_data += extract_avg_ft(rs, mutable_attr, "RO_StatPar")

    rs = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], metric_name='Equal opportunity difference'))
    plot_data += extract_avg_ft(rs, mutable_attr, "RO_EqOpp")

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])


    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.show()
    plt.savefig("notion_fairness_comparision.png")

# Compare different times of intervention
if False:
    learners = [("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group])),
        ("in",FairLearnLearner([privileged_group], [unprivileged_group])),
        ("post",RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))]

    y_attr = mutable_attr
    for name, l in learners:
        rs = do_sim(l)
        plot_data += extract_avg_ft(rs, mutable_attr, name)

    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", y_attr])

    sns.catplot(x="name", y=y_attr, hue="time", kind="bar",
                data=plot_data_df)
    plt.show()
    plt.savefig("stat_parity_times.png")

# point of intervention
# algorithmic intervention (lower threshold)
## here it might make sense for metrics != statpar (e.g. groundtruth)
## how different 'strengths' of interventions (threshold boost) affect result
# vs. pre-intervention (lower cost of manipulation)
## group dependant cost function... allow negative for incentive boost...
metric_name = 'mutablediff'
subsidies = np.linspace(0,1,4)
learner = LogisticLearner(exclude_protected=True)

# creates a instance dependant cost lambda fn
# which reduces cost for unprivileged group by subsidy
def cost_fixed_dependant(subsidy):
    group_attr_ind = data._ft_index(group_attr)
    privileged_val = privileged_group[group_attr]

    return lambda x: cost_fixed(1)[0] if x[group_attr_ind] == privileged_val else cost_fixed(1)[0] - subsidy

result_sets = list(map(lambda s: do_sim(learner, cost_fixed_dep=cost_fixed_dependant(s)), subsidies))
result_sets += list(map(lambda s: do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], abs_bound=s)), subsidies))

y_values = list(map(lambda rs: extract_metric(rs, metric_name), result_sets))
index = ['pre'] * len(subsidies) + ['post'] * len(subsidies)
x = np.array([np.hstack((subsidies,subsidies)), y_values]).transpose()

plot_data_df = pd.DataFrame(x, columns=['subsidy', metric_name],index=index).reset_index()
print(plot_data_df)
ax = sns.lineplot(x='subsidy',hue='index', y=metric_name, data=plot_data_df)
plt.show()

#for rs in result_sets:
#    ax = sns.lineplot(x=mutable_attr, y="incentive",hue=group_attr,data=(rs.                     _avg_incentive(mutable_attr, group_attr)).reset_index())
#    plt.show()
