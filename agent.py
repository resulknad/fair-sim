import time

class RationalAgent:
    def __init__(self, h, dataset, cost, x, y):
        self.dataset = dataset
        self.cost_fixed = cost
        self.x = x
        self.y = y
        self.h = h

    def benefit(self, x_new):
        return self.h(x_new)

    def cost(self, x_new):
        assert(self.cost_fixed>=0)
        return self.cost_fixed +self.dataset.dynamic_cost(x_new, self.x)
##
    def incentive(self, x_new):
        inc = self.benefit(x_new) - self.cost(x_new)
        return inc
