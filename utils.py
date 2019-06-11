from aif360.datasets import BinaryLabelDataset
import pandas as pd
import numpy as np
from functools import reduce

def dataset_from_matrix(x, dataset):
    df = pd.DataFrame(data=x, columns=dataset.feature_names + dataset.label_names)
    dataset_ = BinaryLabelDataset(df=df, label_names=dataset.label_names, protected_attribute_names=dataset.protected_attribute_names)

    dataset_ = dataset.align_datasets(dataset_)
    #dataset_.favorable_label = dataset.favorable_label
    dataset_.validate_dataset()
    return dataset_

def _df_selection(df, selection_criteria):
    if len(selection_criteria.items()) == 0:
        return df
    # ands all the selection criterias, returns selected rows
    arr = list(map(lambda tpl: np.array(df[tpl[0]] == tpl[1]), selection_criteria.items()))
    return df[reduce(lambda x,y: x&y, arr)]

def count_df(df, selection_criterias):
    return np.array(list(map(lambda x: len(_df_selection(df,x)), selection_criterias)))
