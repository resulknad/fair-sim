import numpy as np
from numpy.random import normal, binomial

from featuredescription import FeatureDescription
from simulation import Simulation
from learner import RegressionLearner
from agent import AgentContinious
SIZE = 1000
X = {"ft1": normal(loc=1.0, scale=100, size=SIZE), "ft2": normal(loc=1.0, scale=100, size=SIZE), "mutable": normal(loc=20.0, scale=100, size=SIZE), "group": np.random.binomial(1,0.5,SIZE)}

# y is lin comb of features + some gaussian noise
Y = [x[0] + 2*x[1] + 5*x[2] + x[3] for x in zip(X['ft1'], X['ft2'], X['mutable'], normal(loc=0, scale=10, size=SIZE))]


feature_desc = FeatureDescription(X, Y)
feature_desc.add_descr('mutable', mutable=True, domain=None, cost_fn=lambda x_new, x: pow(abs(x_new-x),1.2))
feature_desc.add_descr('group', protected=True, mutable=False, group=True)
feature_desc.add_descr('ft1', mutable=False)
feature_desc.add_descr('ft2', mutable=False)

sim = Simulation(feature_desc, AgentContinious, RegressionLearner, lambda size: np.random.normal(loc=1.0,size=size))

sim.start_simulation(include_protected=True)

sim.plot_mutable_features()

#sim.plot_group_y('pre')
#sim.plot_group_y('post')

sim.show_plots()
