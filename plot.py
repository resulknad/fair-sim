import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex

from itertools import starmap
import numpy as np
from functools import reduce
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from utils import _df_selection, count_df


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


def merge_dfs(col, colval1, colval2, df1, df2):
    df1[col] = pd.Series([colval1] * len(df1.index), df1.index, dtype="category")
    df2[col] = pd.Series([colval2] * len(df2.index), df2.index, dtype="category")
    return pd.concat((df1, df2), ignore_index=True)

def merge_all_dfs(result_set):
    df = pd.concat(list(map(lambda r: r.df, result_set.results)), ignore_index=True)
    df_new = pd.concat(list(map(lambda r: r.df_new, result_set.results)), ignore_index=True)
    return df, df_new

def modify_legend(labels):
    search = ['0_0', '0_1', '1_0', '1_1']
    replace = ['Initial UP', 'Impacted UP', 'Initial P', 'Impacted P']
    for i in range(len(labels)):
        for k,v in zip(search, replace):
            labels[i] = labels[i].replace(k,v)
    return labels

def print_logreg_coeffs(data):
    l = LogisticLearner()
    l.fit(data)

    display(Markdown("#### LogReg Coeffs."))
    display(pd.DataFrame(columns=['Feature', 'Coefficient LogReg'], data=l.coefs))

def plot_ga(rs, index, features):
    # benefit, cost, incentive_mean graph
    d = rs.results[0].incentives
    #np.argmax(np.array(d[0][0])[:,3] - np.array(d[len(d)-1][0])[:,3])
    feature_ind = list(map(d[0]['names'].index, features))
    extracted_features = np.array(list(map(lambda ft_ind: list(starmap(lambda i,x: [i, np.mean(x['features'][:,ft_ind])], zip(range(len(d)), d))), feature_ind))).reshape(-1,2)

    benefit = list(starmap(lambda i,x: [i, np.mean(x['benefit'][index])], zip(range(len(d)), d)))
    boost = list(starmap(lambda i,x: [i, x['boost']], zip(range(len(d)), d)))
    incentive_mean = list(starmap(lambda i,x: [i, np.mean(x['benefit'])-np.mean(x['cost'])], zip(range(len(d)), d)))
    cost = list(starmap(lambda i,x: [i, np.mean(x['cost'][index])], zip(range(len(d)), d)))

    indx = np.array(list(map(lambda a: [a]*len(d), [*features,'benefit']))).ravel()
    df = pd.DataFrame(data=(np.vstack((extracted_features,
                                       benefit))),
                                       #cost))),
                                       #incentive_mean,
                                       #boost))),
                      columns=["t", "val"],
                      index=(indx)).reset_index()
                             #+ ["cost"] * len(d))).reset_index()
                             #+ ["incentive_mean"] * len(d)
                             #+ ["boost"] * len(d))).reset_index()
    plt.figure()
    ax = sns.lineplot(x='t', y="val", hue='index',data=df)
    plt.show()


def plot_distribution(dataset, dist_plot_attr):
    dataset.infer_domain()
    fns = dataset.rank_fns()
    sample = np.linspace(-1,1,100)
    data_arr = list(map(fns[0][dist_plot_attr], sample))
    data_arr.extend(list(map(fns[1][dist_plot_attr], sample)))
    data_arr = np.array([np.hstack((sample,sample)), data_arr]).transpose()
    df = pd.DataFrame(data=data_arr, columns=['x', 'y'])
    ax = sns.lineplot(x='x', y="y",data=df)
    display(Markdown("### Distribution of " + dist_plot_attr))
    plt.show()

def plot_all_mutable_features(rs, unprivileged_group, privileged_group, dataset,all_mutable,name='a', kind='pdf'):
    sns.set_style("whitegrid")

    ft_name = 'credit_h_pr'


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
        merged = merged.sort_values(mutable_attr)


        palette = {'0_0': '#4286f4', '0_1':'#91bbff', '1_0':'#f45942', '1_1':'#ff9282'}
        ylabel = 'probability density'

        if np.issubdtype(merged[mutable_attr], np.number):
            if kind == 'cdf':
                ylabel = 'cumulative probability'
                for time in ['0_0', '0_1', '1_0','1_1']:
                    mask = merged['time'] == time
                    merged.loc[mask,'index'] = merged.loc[mask,'index'].cumsum()
            ax = sns.pointplot(scale=0.75,x=mutable_attr, hue="time", y="index",
                        data=merged, palette=palette, linestyles=['-','--','-','--'])
        else:
            ax = sns.barplot(x=mutable_attr, hue="time", y="index",
                        data=merged,  palette=palette)
        handles, labels = ax.get_legend_handles_labels()
        #print(ax.get_xlabel())

        ax.set(ylabel=ylabel, xlabel=ax.get_xlabel().replace('_', ' '))
        ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        ax.legend(handles=handles[0:], labels=modify_legend(labels[0:]))
        pp = PdfPages('figures/' + name + '_' + ax.get_xlabel() +'.pdf')
        pp.savefig(bbox_inches="tight")
        pp.close()
        plt.show()

def merge_result_sets(rss, unprivileged_group, privileged_group, ft_name):
    plot_data = pd.DataFrame(data=[],columns=["name", "time", ft_name])

    for name, rs in rss:
        for sc in [unprivileged_group, privileged_group]:
            df, df_post = merge_all_dfs(rs)
            df = _df_selection(df, sc)
            df_post = _df_selection(df_post, sc)

            grp = str(list(sc.values())[0])
            merged_df = merge_dfs('time', grp + '_0', grp + '_1', df, df_post)
            merged_df = merged_df[['time', ft_name]]
            merged_df['name'] = pd.Series([name] * len(merged_df.index), merged_df.index)
            plot_data = pd.concat((plot_data, merged_df), ignore_index=True)
    plot_data_df = pd.DataFrame(plot_data, columns=["name", "time", ft_name])
    return plot_data_df

def boxplot(rss, up, p, name=''):
    ft_name = 'credit_h_pr'
    plot_data_df = merge_result_sets(rss, up, p, ft_name)
    sns.set_style("whitegrid")
    palette = {'0_0': '#4286f4', '0_1':'#91bbff', '1_0':'#f45942', '1_1':'#ff9282'}

    ax = sns.boxplot(x="name", y=ft_name, hue="time",
                data=plot_data_df, palette=palette)




    pp = PdfPages('figures/' + name + '.pdf')
    handles, labels = ax.get_legend_handles_labels()
    print(ax.get_xlabel())
    ax.set(ylabel='benefit', xlabel=ax.get_xlabel().replace('_', ' '))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=0)
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0., handles=handles[0:], labels=modify_legend(labels[0:]))

    pp.savefig(bbox_inches="tight")
    pp.close()
    plt.show()
