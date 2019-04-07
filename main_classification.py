from pynverse import inversefunc
import itertools
from sklearn.preprocessing import KBinsDiscretizer

import numpy as np
import pandas
from scipy.stats import norm

from featuredescription import FeatureDescription
from simulation import Simulation
from learner import LogisticLearner
from agent import AgentDiscrete


# dataset specific preprocessing
a = pandas.read_csv("german.data.bak", delim_whitespace=True)

categorical = ['balance', 'statusandsex', 'job', 'foreigner', 'credithistory', 'purpose', 'creditamount', 'savings', 'employmentsince', 'otherdebitors', 'residencesince', 'property', 'housing', 'telephone', 'otherinstallment']
numerical = ['duration', 'installmentrate', 'age', 'existingcredits', 'dependants']

def tti(x): #text to index
    distinct_elements = list(set(x))
    distinct_elements.sort()
    # print(set(map(lambda el: str(el) + " -> " + str(distinct_elements.index(el)), x)))

    return list(map(lambda el: distinct_elements.index(el), x))

def nti(x): #numeric to index
    xt = np.matrix([x]).transpose()
    return list(map(lambda x:int(x[0]),KBinsDiscretizer(encode='ordinal', n_bins=10).fit_transform(xt)))



X_c = {k: tti(a[k]) for k in categorical}
X_n = {k: nti(a[k]) for k in numerical}

X = {**X_c, **X_n}
Y = list(map(lambda x: 2-x, a['y']))

x = tti(a['balance'])
acc = {}
for bal, yy in zip(x,Y):
    if (yy,bal) not in acc:
        acc[(yy,bal)] = 0
    acc[(yy,bal)] += 1

print(acc)

# https://newonlinecourses.science.psu.edu/stat508/lesson/gcd/gcd.2
# removed some attributes according to pearson chi square test

feature_desc = FeatureDescription(X, Y)
feature_desc.add_descr('balance', mutable=True, domain=list(range(0,4)))
feature_desc.add_descr('statusandsex', protected=True, mutable=False)
feature_desc.add_descr('foreigner', protected=True, mutable=False, group=True)
# feature_desc.add_descr('job', domain=list(range(0,4)), mutable=False)
feature_desc.add_descr('duration', domain=list(range(0,10)), mutable=False)

feature_desc.add_descr('credithistory', domain=list(range(0,len(set(a['credithistory'])))), mutable=False)
feature_desc.add_descr('purpose', domain=list(range(0,len(set(a['purpose'])))), mutable=False)
feature_desc.add_descr('creditamount', domain=list(range(0,len(set(a['creditamount'])))), mutable=False)
feature_desc.add_descr('savings', domain=list(range(0,len(set(a['savings'])))), mutable=True)
feature_desc.add_descr('employmentsince', domain=list(range(0,len(set(a['employmentsince'])))), mutable=False)
# feature_desc.add_descr('installmentrate', domain=list(range(0,len(set(a['installmentrate'])))), mutable=False)
feature_desc.add_descr('otherdebitors', domain=list(range(0,len(set(a['otherdebitors'])))), mutable=False)
# feature_desc.add_descr('residencesince', domain=list(range(0,len(set(a['residencesince'])))), mutable=False)
feature_desc.add_descr('property', domain=list(range(0,len(set(a['property'])))), mutable=False)
feature_desc.add_descr('age', domain=list(range(0,len(set(a['age'])))), mutable=False)
# feature_desc.add_descr('otherinstallment', domain=list(range(0,len(set(a['otherinstallment'])))), mutable=False)
feature_desc.add_descr('housing', domain=list(range(0,len(set(a['housing'])))), mutable=False)
feature_desc.add_descr('existingcredits', domain=list(range(0,len(set(a['existingcredits'])))), mutable=False)
# feature_desc.add_descr('dependants', domain=list(range(0,len(set(a['dependants'])))), mutable=False)
# feature_desc.add_descr('telephone', domain=list(range(0,len(set(a['telephone'])))), mutable=False)


sim = Simulation(feature_desc, AgentDiscrete, LogisticLearner, lambda size: np.random.normal(loc=1.0,size=size))

sim.start_simulation(include_protected=True)

sim.plot_mutable_features()
sim.plot_group_y('pre')
sim.plot_group_y('post')

sim.show_plots()
