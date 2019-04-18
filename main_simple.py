from mutabledataset import SimpleDataset
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner
import plot

import numpy as np

from learner import StatisticalParityLogisticLearner
cost_lambda = lambda x_new, x: x_new/2.+1*abs(x_new-x)/4.
cost_fixed = lambda size: np.abs(np.random.normal(loc=0.5,size=size))


g = SimpleDataset(mutable_features=['x'],
        domains={'x': 'auto'},
        discrete=['x'],
        cost_fns={'x': lambda x_new, x: pow(x_new/2.,2.)+3*abs(x_new-x)/4.})
#x_new/2.+1*
privileged_groups = [{'group': 1}]
unprivileged_groups = [{'group': 0}]


from learner import EqOddsPostprocessingLogisticLearner

g = SimpleDataset(mutable_features=[],
        domains={'x': 'auto'},
        discrete=['x'],
        cost_fns={'x': cost_lambda})

privileged_groups = [{'group': 1}]
unprivileged_groups = [{'group': 0}]

sim = Simulation(g,
                 RationalAgent,
                 EqOddsPostprocessingLogisticLearner(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups),
                 cost_fixed)
print(sim.dataset.unfavorable_label)
sim.start_simulation(include_protected=True)

