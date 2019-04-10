from mutabledataset import GermanSimDataset
from agent import RationalAgent
from simulation import Simulation
from learner import LogisticLearner
import numpy as np

g = GermanSimDataset(mutable_features=['status', 'savings'], domains={'savings': 'auto', 'status': 'auto'}, discrete=['status', 'savings'])

sim = Simulation(g, RationalAgent, LogisticLearner, lambda size: np.abs(np.random.normal(loc=1.0,size=size)))

sim.start_simulation(include_protected=True)

sim.plot_mutable_features()
sim.plot_group_y('pre')
sim.plot_group_y('post')

sim.show_plots()

