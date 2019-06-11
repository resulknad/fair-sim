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
from learner import RandomForestLearner, GaussianNBLearner
from learner import RejectOptionsLogisticLearner
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
all_mutable = mutable_monotone_pos + mutable_monotone_neg + mutable_dontknow #+ categorical
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


# -

# ## LogReg feature coefficients

# +
from matplotlib.backends.backend_pdf import PdfPages

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

def merge_dfs(col, colval1, colval2, df1, df2):
    df1[col] = pd.Series([colval1] * len(df1.index), df1.index, dtype="category")
    df2[col] = pd.Series([colval2] * len(df2.index), df2.index, dtype="category")
    return pd.concat((df1, df2), ignore_index=True)

def merge_all_dfs(result_set):
    df = pd.concat(list(map(lambda r: r.df, result_set.results)), ignore_index=True)
    df_new = pd.concat(list(map(lambda r: r.df_new, result_set.results)), ignore_index=True)
    return df, df_new

def plot_all_mutable_features(rs,name='a'):
    pp = PdfPages(name + '.pdf')

    all_mutable_dedummy = list(set(map(lambda s: s.split('=')[0], all_mutable)))

    df, df_post = merge_all_dfs(rs)
    df = df.replace(dataset().human_readable_labels)
    df_post = df_post.replace(dataset().human_readable_labels)

    N = count_df(df, [unprivileged_group, privileged_group])

    for mutable_attr in all_mutable_dedummy:
        dfs = []
        for sc in [unprivileged_group, privileged_group]:

            df_ = _df_selection(df, sc)

            df_post_ = _df_selection(df_post, sc)

            grp = str(list(sc.values())[0])
            dfs.append(merge_dfs('time', grp + '_0' , grp+'_1', df_, df_post_))
        merged = pd.concat(dfs)

        #merged = merge_dfs('time', 'pre', 'post', df, df_new)
        merged = merged.reset_index(drop=True).reset_index().groupby([mutable_attr, 'time']).count().reset_index()

        def normalize(row):
            if row['time'][0] == '0':
                row['index'] /= N[0]
            else:
                row['index'] /= N[1]
            return row

        merged = merged.apply(normalize, axis=1)

        #display(merged)
        plt.figure(mutable_attr)
        merged['time'] = merged['time'].astype('category')

        # if datapoint is missing, there's a gap
        # we don't want that
        for t in merged[mutable_attr]:
            for h in list(set(merged['time'])):
                if (((merged['time'] == h) & (merged[mutable_attr] == t)).sum()) == 0:
                    merged = merged.append({'time': h, mutable_attr: t, 'index': 0.}, ignore_index=True)
                    # datapoint is missing
                    # add one with y=0
        #print(merged.dtypes)
        palette = {'0_0': '#4286f4', '0_1':'#91bbff', '1_0':'#ffab28', '1_1':'#ffd491'}
        if np.issubdtype(merged[mutable_attr], np.number):
            print(mutable_attr)
            sns.pointplot(x=mutable_attr, hue="time", y="index",
                        data=merged, palette=palette, linestyles=['-','--','-','--'])
        else:
            sns.barplot(x=mutable_attr, hue="time", y="index",
                        data=merged,  palette=palette)
        pp.savefig()

        plt.show()
    pp.close()

if False:
    data = dataset()
    #rs = do_sim(FairLearnLearner([privileged_group], [unprivileged_group]), collect_incentive_data=True)
    rs = do_sim(RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]), collect_incentive_data=True)
    #rs = do_sim(LogisticLearner(exclude_protected=True), collect_incentive_data=True)
    # benefit, cost, incentive_mean graph
    d = rs.results[0].incentives
    index = 147 #np.argmax(np.array(d[0][0])[:,3] - np.array(d[len(d)-1][0])[:,3])
    savings = list(starmap(lambda i,x: [i, np.mean(x[0][:,1])], zip(range(len(d)), d)))
    benefit = list(starmap(lambda i,x: [i, np.mean(x[1])], zip(range(len(d)), d)))
    incentive_mean = list(starmap(lambda i,x: [i, np.mean(x[1])-np.mean(x[2])], zip(range(len(d)), d)))
    cost = list(starmap(lambda i,x: [i, np.mean(x[2])], zip(range(len(d)), d)))
    df = pd.DataFrame(data=(np.vstack((savings,benefit,cost,incentive_mean))), columns=["t", "val"], index=(["month"]*len(d) + ["benefit"]*len(d) + ["cost"] * len(d)+ ["incentive_mean"] * len(d))).reset_index()
    plt.figure()
    ax = sns.lineplot(x='t', y="val", hue='index',data=df)
    plt.show()
    plot_all_mutable_features(rs)

#data = dataset()
#rs = do_sim(LogisticLearner(exclude_protected=True), collect_incentive_data=False)
#plot_all_mutable_features(rs)


# -

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


# # Notions of Fairness

# +
# Compare different notions of fairness

data = dataset()
learners = [("baseline", LogisticLearner(exclude_protected=True)),
            ("RO_StatPar", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group])),
            ("RO_EqOpp", RejectOptionsLogisticLearner([privileged_group], [unprivileged_group], metric_name='Equal opportunity difference'))]
#
ft_name = 'credit_h_pr'
plot_data = pd.DataFrame(data=[],columns=["name", "time", ft_name])

for name, l in learners:
    rs = do_sim(l)
    display(Markdown("## " + name))
    plot_all_mutable_features(rs,name=name)

    for sc in [unprivileged_group, privileged_group]:
        df, df_post = merge_all_dfs(rs)
        df = _df_selection(df, sc)
        df_post = _df_selection(df_post, sc)

        grp = str(list(sc.values())[0])
        merged_df = merge_dfs('time', 'pre' + grp, 'post'+ grp, df, df_post)
        merged_df = merged_df[['time', ft_name]]
        merged_df['name'] = pd.Series([name] * len(merged_df.index), merged_df.index)
        plot_data = pd.concat((plot_data, merged_df), ignore_index=True)
        #plot_data.append([name, 'pre', extract_metric(rs, metric_name=metric_name, time='pre')])
        #plot_data.append([name, 'post', extract_metric(rs, metric_name=metric_name, time='post')])

plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", ft_name])

plt.figure()
sns.boxplot(x="name", y=ft_name, hue="time",
            data=plot_data_df)
plt.show()

# average benefit
# interval plot
# foreach notion of fairness, each mutable features
# -



# ## Statistical parity comparison
# with protected attributes included, otherwise it's boring...
#
# gtdiff: difference of averages (of priv and unpriv) of the new ground truth (assigned by KNN with custom cost function as distance)

# +
data = dataset()
plot_data = []
learners = [("baseline", LogisticLearner(exclude_protected=False)),
    ("pre", ReweighingLogisticLearner([privileged_group], [unprivileged_group])),
    ("in",FairLearnLearner([privileged_group], [unprivileged_group])),
    ("post",RejectOptionsLogisticLearner([privileged_group], [unprivileged_group]))]

#metric_name = 'gtdiff'


ft_name = 'credit_h_pr'
plot_data = pd.DataFrame(data=[],columns=["name", "time", ft_name])

for name, l in learners:
    rs = do_sim(l)
    display(Markdown("## " + name))
    #plot_all_mutable_features(rs)

    for sc in [unprivileged_group, privileged_group]:
        df, df_post = merge_all_dfs(rs)
        df = _df_selection(df, sc)
        df_post = _df_selection(df_post, sc)

        grp = str(list(sc.values())[0])
        merged_df = merge_dfs('time', 'pre' + grp, 'post'+ grp, df, df_post)
        merged_df = merged_df[['time', ft_name]]
        merged_df['name'] = pd.Series([name] * len(merged_df.index), merged_df.index)
        plot_data = pd.concat((plot_data, merged_df), ignore_index=True)
        #plot_data.append([name, 'pre', extract_metric(rs, metric_name=metric_name, time='pre')])
        #plot_data.append([name, 'post', extract_metric(rs, metric_name=metric_name, time='post')])

plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", ft_name])

plt.figure()
sns.boxplot(x="name", y=ft_name, hue="time",
            data=plot_data_df)
plt.show()


#for name, l in learners:
#    rs = do_sim(l)
#    plot_data.append([name, 'pre', extract_metric(rs, metric_name=metric_name, time='pre')])
#    plot_data.append([name, 'post', extract_metric(rs, metric_name=metric_name, time='post')])

#plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", metric_name])

#plt.figure()
#sns.catplot(x="name", y=metric_name, hue="time", kind="bar",
#            data=plot_data_df)
#plt.show()

# split for groups
# 1. section 2.1 for cost function
# 2. cost function as distance, only within group (immutable)
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


