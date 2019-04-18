import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from aif360.datasets import BinaryLabelDataset

import re

from transformer import AgentTransformer

class Simulation(object):
    def __init__(self, dataset, AgentCl, learner, cost_distribution):
        self.dataset = dataset
        self.cost_distribution = cost_distribution
        self.learner = learner
        self.AgentCl = AgentCl
        self.scaler = min_max_scaler = MaxAbsScaler()

    def no_classes(self, dataset):
        return len(set(dataset.labels.ravel()))

    @staticmethod
    def dataset_from_matrix(x, dataset):
        df = pd.DataFrame(data=x, columns=dataset.feature_names + dataset.label_names)
        dataset_ = BinaryLabelDataset(df=df, label_names=dataset.label_names,              protected_attribute_names=dataset.protected_attribute_names)

        dataset_ = dataset.align_datasets(dataset_)
        #dataset_.favorable_label = dataset.favorable_label
        dataset_.validate_dataset()
        return dataset_



    def start_simulation(self, include_protected=True):
        # we need at least one example for each class in each of the two splits
        while True:
            train,test = self.dataset.split(2, shuffle=True)
            break
            if self.no_classes(train) >= 2 and self.no_classes(test) >= 2:
                break
        train_indices = list(map(int, train.instance_names))
        test_indices = list(map(int, test.instance_names))

        print("Train: ",train.features.shape,", Test: ", test.features.shape)
        self.train, self.test = train,test
        train.features = self.scaler.fit_transform(train.features)
        test.features = self.scaler.transform(test.features)
        self.dataset.features = self.scaler.transform(self.dataset.features)
        self.dataset.infer_domain()

        #t = self.dataset.features
        #rows = np.where(t[:,1] == 0.0)
        #result = t


        # learner moves
        h = self.learner.fit(train)
        self.Y_predicted = h(self.dataset.features, False)

        # agents move
        at = AgentTransformer(self.AgentCl, h, self.cost_distribution, self.scaler)

        dataset_ = at.transform(self.dataset)
        train_ = Simulation.dataset_from_matrix(np.hstack((dataset_.features[train_indices,:], dataset_.labels[train_indices])),self.dataset)
        test_ = Simulation.dataset_from_matrix(np.hstack((dataset_.features[test_indices,:], dataset_.labels[test_indices])),self.dataset)


        print("Accuracy (h) pre",self.learner.accuracy(test))


        # update changed features

        #dataset_ = self.dataset_from_matrix(np.hstack((np.vstack((train_.features, test_.features)), np.vstack((train_.labels, test_.labels)))), self.dataset)
        self.Y_new_predicted = h(dataset_.features, False)

        acc_h_post = self.learner.accuracy(test_)
        print("Accuracy (h) post",acc_h_post)

        # fit data again, see if accuracy changes
        self.learner.fit(train_)
        acc_h_star_post = self.learner.accuracy(test_)
        print("Accuracy (h*) post",acc_h_star_post)
        print("eps = ",round(abs(acc_h_star_post-acc_h_post),2))
        print("y=1",sum((train_.labels + test_.labels).ravel())," <- ", sum((train.labels + test.labels).ravel()))

        self.dataset_new = dataset_


        self.dataset_df = self.dataset.convert_to_dataframe(de_dummy_code=True)[0]
        self.dataset_df['credit_h'] = pd.Series(self.Y_predicted, index=self.dataset_df.index)
        self.dataset_new_df = self.dataset_new.convert_to_dataframe(de_dummy_code=True)[0]
        self.dataset_new_df['credit_h'] = pd.Series(self.Y_new_predicted, index=self.dataset_new_df.index)
