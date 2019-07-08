import time
import numpy as np

class RationalAgent:
    """Describing the behavior of agents, namely their `utility`, `cost` and `benefit` for specific manipulations.

    :param h: Trained instance of some `learner`, supporting `predict` and `predict_proba`.
    :param dataset: Some dataset extending `SimMixin`
    :param cost: Function that calculates the cost of manipulation
    :param X: Feature matrix
    :param y: Labels"""
    threshold = 0.
    def __init__(self, h, dataset, cost, X, y):
        self.dataset = dataset
        self.cost_fixed = cost
        self.X = X
        self.X_benefit = np.array(h.predict_proba(X))
        self.y = y
        self.h = h

    def benefit(self, X_new):
        """Returns benefit of manipulation from initial `X` to `X_new`. In this case this is simply the difference of scores as returned by the predictor `h.predict_proba`.

        :param X_new: Manipulated feature matrix.
        :returns: Vector of benefits"""
        #b = self.h.predict_proba(X_new)
        b = np.array(self.h.predict_proba(X_new)) - self.X_benefit
        return b

    def cost(self, X_new):
        """Returns cost of manipulation. The cost functions passed to the `Simulation` class are used here.

        :param X_new: Manipulated feature matrix.
        :returns: Vector of costs"""
        #print(np.mean(self.cost_fixed[np.where(X_new[:,8]==0)]))
        return np.add(self.cost_fixed,self.dataset.dynamic_cost(X_new, self.X))

    def incentive(self, X_new):
        """Calculates incentive (or utility) of manipulations. In this instance this is simply the difference of benefit and cost.

        :param X_new: Manipulated feature matrix.
        :returns: Vector of incentives"""
        inc = np.add(self.benefit(X_new), - self.cost(X_new))
        return inc

class RationalAgentOrig(RationalAgent):
    threshold = 0.
    def benefit(self, X_new):
        if self.threshold == 0.:
            raise Warning("threshold 0")
        pr = np.array(self.h.predict_proba(X_new))
        b =  np.clip(pr/(self.threshold+0.2), None,1.)
        return b

        # b = np.clip(pr, None, 0.6) + np.interp(pr - 0.4, [0,0.1], [0,0.4])
        # return b
