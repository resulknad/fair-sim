from pynverse import inversefunc
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

class Agent(object):
    def __init__(self, cost, x, y):
        self.cost_fixed = cost
        self.x = np.array(x).flatten()

    def benefit(self, h, x_new):
        return h(x_new) - h(self.x)

    def cost(self, x_new):
        return self.cost_fixed + 0.001*np.abs(x_new-self.x)

    def incentive(self, h, x_new):
        return self.benefit(h,x_new) - self.cost(x_new)

    def act(self, h):
        utility = lambda x: -self.incentive(h, x)

        # limit x
        # cons = ({'type': 'ineq', 'fun': lambda x: 20-np.abs(self.x-x)}) , constraints=cons
        res = minimize(utility, self.x)

        if utility(res.x) < 0:
            print(res.x,self.x,utility(res.x))
            self.x = res.x

class Learner(object):

    def fit(self, X, y):
        reg = LinearRegression().fit(X, y)
        return lambda x: reg.predict([x])
        # returns h

class Simulation(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.plot("pre")

        # draw cost
        self.X_cost = np.random.standard_exponential(X.shape[0])

        # learner moves
        self.learner = Learner()
        h = self.learner.fit(X,Y)

        # agents learn about cost
        self.agents = [Agent(cost, x, y) for (cost, x, y) in zip(self.X_cost, X, Y)]
        for a in self.agents:
            a.act(h)

        self.X = np.matrix([a.x for a in self.agents])
        print(self.X)
        self.plot("post")

        # plot h
        xvals = np.linspace(-100,100,1000) #100 points from 0 to 6 in ndarray
        yvals = list(map(lambda x: h([x]), xvals)) #evaluate f for each point in xvals
        plt.figure("h")
        plt.plot(xvals, yvals)

        plt.show()



    def plot(self, text):
        plt.figure(text)
        n, bins, patches = plt.hist(x=self.X, bins='auto', color='#0504aa',
                                    alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(text)
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
        plt.show(block=False)






np.random.seed(0)
SIZE = 100

X = np.transpose(np.matrix(np.random.standard_normal(SIZE)))
y = np.random.standard_exponential(SIZE)
sim = Simulation(X, y)

