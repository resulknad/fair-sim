from mutabledataset import GermanSimDataset
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner

import plot
import numpy as np
import pandas as p
from learner import AdversialDebiasingLogisticLearner

g = GermanSimDataset(mutable_features=['status', 'savings'],
                     domains={'savings': 'auto', 'status': 'auto'},
                     discrete=['status', 'savings'],
                     protected_attribute_names=['age'],
                     privileged_classes=[lambda x: x >= 25],
                     features_to_drop=['personal_status', 'sex'])

privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

sim = Simulation(g,
                 RationalAgent,
                 AdversialDebiasingLogisticLearner(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups),
                 lambda size: np.abs(np.random.normal(loc=0.5,size=size)))

sim.start_simulation(include_protected=True)
