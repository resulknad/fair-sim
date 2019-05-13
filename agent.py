import time
import numpy as np

class RationalAgent:
    def __init__(self, h, dataset, cost, X, y):
        self.dataset = dataset
        self.cost_fixed = cost
        self.X = X
        self.y = y
        self.h = h

    def benefit(self, X_new):
        #b = self.h.predict_proba(X_new)
        b = self.h.predict_proba(X_new)
        return b

    def cost(self, X_new):
        return np.add(self.cost_fixed,self.dataset.dynamic_cost(X_new, self.X))

    def incentive(self, X_new):
        inc = np.add(self.benefit(X_new), - self.cost(X_new))
        return inc

class RationalAgentOrig(RationalAgent):
    def benefit(self, X_new):
        pr = np.array(self.h.predict_proba(X_new))
        b = np.clip(pr, None, 0.6) + np.interp(pr - 0.4, [0,0.1], [0,0.4])
        return b
