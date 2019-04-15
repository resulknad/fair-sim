import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import cProfile
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

    def start_simulation(self, include_protected=True):
        # we need at least one example for each class in each of the two splits
        while True:
            train,test = self.dataset.split(2, shuffle=True)
            break
            if self.no_classes(train) >= 2 and self.no_classes(test) >= 2:
                break

        print("Train: ",train.features.shape,", Test: ", test.features.shape)
        self.train, self.test = train,test
        train.features = self.scaler.fit_transform(train.features)
        test.features = self.scaler.transform(test.features)

        # learner moves
        h = self.learner.fit(train)

        self.Y_predicted = h(self.scaler.transform(self.dataset.features), False)

        # agents move
        at = AgentTransformer(self.AgentCl, h, self.cost_distribution)
        train_ = at.transform(train)
        test_ = at.transform(test)

        print("Accuracy (h) pre",self.learner.accuracy(test))

        # update changed features
        dataset_ = self.dataset.copy()
        dataset_.features = np.vstack([train_.features, test_.features])
        print(dataset_.features.shape)
        dataset_.labels = np.vstack([train_.labels, test_.labels])

        self.Y_new_predicted = h(dataset_.features, False)

        print("Accuracy (h) post",self.learner.accuracy(test_))

        # fit data again, see if accuracy changes
        self.learner.fit(train_)
        print("Accuracy (h*) post",self.learner.accuracy(test_))
        print("y=1",sum((train_.labels + test_.labels).ravel())," <- ", sum((train.labels + test.labels).ravel()))



        self.dataset_new = dataset_

        self.dataset_df = self.dataset.convert_to_dataframe(de_dummy_code=True)[0]
        self.dataset_df['credit_h'] = pd.Series(self.Y_predicted, index=self.dataset_df.index)
        self.dataset_new_df = self.dataset_new.convert_to_dataframe(de_dummy_code=True)[0]
        self.dataset_new_df['credit_h'] = pd.Series(self.Y_new_predicted, index=self.dataset_new_df.index)
