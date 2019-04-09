from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import cross_val_score

class LogisticLearner(object):
    def fit(self, X, y):
        #print(cross_val_score(MultinomialNB(), X, y, cv=5))
        #exit()
        reg = MultinomialNB().fit(X,y)
        #reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1.0).fit(X, y)
        self.h = reg

        return lambda x,single=True: reg.predict(x)[0] if single else reg.predict(x)

    def accuracy(self, X, y):
        return self.h.score(X,y)

class RegressionLearner(object):
    def fit(self, X, y):
        reg = LinearRegression().fit(X, y)
        self.h = reg
        return lambda x: reg.predict(x)

    def accuracy(self, X, y):
        return self.h.score(X,y)
