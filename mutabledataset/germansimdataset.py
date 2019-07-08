import itertools
import warnings
import pandas as pd
import numpy as np

from aif360.datasets import GermanDataset
from aif360.datasets import StructuredDataset
from aif360.datasets import BinaryLabelDataset
from scipy.stats import percentileofscore
from .simmixin import SimMixin

default_mappings = {
    #'label_maps': [{1.0: 'Good Credit', 0.0: 'Bad Credit'}]#,
    #'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'},
    #                             {1.0: 'Old', 0.0: 'Young'}],
}

def custom_preprocessing(df):
    N_BINS = 32
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}
    df['sex'] = df['personal_status'].replace(status_map)
    df['foreign_worker'] = df['foreign_worker'].replace({'A201':0, 'A202':1})
    df['savings'] = df['savings'].replace({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65':5})
    df['credit_amount'] = (df['credit_amount']/max(df['credit_amount'])).apply(lambda x: round(x*N_BINS)/N_BINS)
    df['has_checking_account'] = df['month'].apply(lambda x: int(not x=='A14'))
    df['status'] = df['status'].replace({'A11': 0, 'A12': 0.5, 'A13': 1, 'A14':0})
    df['month'] = (df['month']/max(df['month'])).apply(lambda x: round(x*N_BINS)/N_BINS)
    df['credit'] = df['credit'].map(lambda x: 2-x)
    return df

class GermanSimDataset(GermanDataset, SimMixin):
    """
    Inherits from AIF360s `GermanDataset` and :py:obj:`mutabledataset.simmixin.SimMixin`. Some additional pre-processing is done here.
    """
    def __init__(self, *args, **kwargs):
        # remove arguments for sim_args constructor
        sim_args_names = ['mutable_features', 'domains', 'cost_fns', 'discrete']
        sim_args = {k: kwargs.pop(k, None) for k in sim_args_names}

        kwargs['custom_preprocessing'] = custom_preprocessing
        kwargs['metadata'] = default_mappings
        kwargs['categorical_features'] = ['credit_history', 'purpose',
                     'employment', 'other_debtors', 'property',
                     'installment_plans', 'housing', 'skill_level', 'telephone'
]

        self.human_readable_labels ={"A40": "car (new)",
            "A41": "car (used)",
            "A42": "furniture/equipment",
            "A43": "radio/television",
            "A44": "domestic appliances",
            "A45": "repairs",
            "A46": "education",
            "A47": "vacation",
            "A48": "retraining",
            "A49": "business",
            "A410": "others",
            "A30": "no credits taken",
            "A31": "all credits at this bank paid back duly",
            "A32": "existing credits paid back duly till now",
            "A33": "delay in paying off in the past",
            "A34": "critical account",
            "A71": "unemployed",
            "A72": "< 1 year",
            "A73": "1  <= ... < 4 years",
            "A74": "4  <= ... < 7 years",
            "A75": ">= 7 years",
            "A101": "none",
            "A102": "co-applicant",
            "A103": "guarantor",
            "A121": "real estate",
            "A122": "building society savings agreement/life insurance",
            "A123": "car or other",
            "A124": "unknown / no property",
            "A141": "bank",
            "A142": "stores",
            "A143": "none",
            "A151": "rent",
            "A152": "own",
            "A153": "for free",
            "A171": "unemployed/ unskilled  - non-resident",
            "A172": "unskilled - resident",
            "A173": "skilled employee / official",
            "A174": "management/ self-employed/ Highly qualified employee/ officer",
            "A191": "none",
            "A192": "yes, registered under the customers name"}

        GermanDataset.__init__(*(tuple([self]) + args), **kwargs)
        SimMixin.__init__(self, **sim_args)
