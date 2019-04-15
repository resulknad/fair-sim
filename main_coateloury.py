from mutabledataset import CoateLouryDataset
from agent import RationalAgent
from simulation import Simulation
from learner import CoateLouryLearner
import plot

import numpy as np

# initial beliefs
pi_b = 0.5
pi_w = 0.6

# utility for learner
x_u = 1
x_q = 2

# as defined in patronizing equilibrium
theta_u = 0.4
theta_q = 0.3

# fns to draw
f_u_draw = lambda: np.random.uniform(0,theta_u,1)[0]
f_q_draw = lambda: np.random.uniform(theta_q,1,1)[0]

# pdfs
f_u = lambda x: 1./theta_u if x >=0 and x<=theta_u else 0.
f_q = lambda x: 1./(1-theta_q) if x >= theta_q and x<=1 else 0.

g = CoateLouryDataset(mutable_features=['educated', 'signal'], domains={'educated': [1], 'signal': f_q_draw}, discrete=['educated', 'signal'], f_u_draw=f_u_draw, f_q_draw=f_q_draw, pi_b=pi_b, pi_w=pi_w)

sim = Simulation(g, RationalAgent, CoateLouryLearner(f_u=f_u, f_q=f_q, x_u=x_u, x_q=x_q, pi_w=pi_w, pi_b=pi_b), lambda size: np.abs(np.random.normal(loc=1.0,size=size)))

print(list(g.discrete_permutations()[1]))
sim.start_simulation(include_protected=True)

plot.plot_mutable_features(sim)
#plot.plot_pie(sim, [{'age': 1}])
plot.block()
