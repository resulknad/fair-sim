from mutabledataset import GermanSimDataset
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner
import plot

import numpy as np


g = GermanSimDataset(mutable_features=['status', 'savings'], domains={'savings': 'auto', 'status': 'auto'}, discrete=['status', 'savings'])

sim = Simulation(g, RationalAgent, LogisticLearner(), lambda size: np.abs(np.random.normal(loc=1.0,size=size)))

sim.start_simulation(include_protected=True)

plot.plot_mutable_features(sim)
#plot.plot_pie(sim, [{'age': 1}])
plot.block()
