import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import pandas as pd


def block():
    plt.show()

def plot_mutable_features(sim):
    df,_ = sim.dataset.convert_to_dataframe(de_dummy_code=True)
    df_new,_ = sim.dataset_new.convert_to_dataframe(de_dummy_code=True)

    disc_and_mutable = sim.dataset._discrete_and_mutable()
    for ft in disc_and_mutable:
        plt.figure()
        n, bins, patches = plt.hist(x=[sorted(df[ft]), sorted(df_new[ft])], label=['pre', 'post'], bins=10,
                                    alpha=0.7, rwidth=0.85)
    #        plt.xlim(left=min(min(data_new),min(data)), right=max(max(data_new),max(data)))
        plt.legend(prop={'size': 10})
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(ft)
        # plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = (n[0] + n[1]).max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

        plt.show(block=False)


def _df_selection(df, selection_criteria):
    # ands all the selection criterias, returns selected rows
    arr = list(map(lambda tpl: np.array(df[tpl[0]] == tpl[1]), selection_criteria.items()))
    return df[reduce(lambda x,y: x&y, arr)]

def count_df(df, selection_criterias):
    return np.array(list(map(lambda x: len(_df_selection(df,x)), selection_criterias)))

def plot_pie(df, selection_criterias, labels, title):
    plt.figure()
    plt.pie(count_df(df, selection_criterias), labels=labels, autopct='%1.1f%%', shadow=True)
    plt.title(title)
