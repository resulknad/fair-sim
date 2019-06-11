from sklearn.naive_bayes import GaussianNB
class GaussianNBLearner(object):
    threshold = 0.5
    def __init__(self, exclude_protected=False):
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        self.dataset = dataset
        reg = GaussianNB().fit(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())

        #print(sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: abs(x[1])))
        #exit(1)
        self.h = reg

    def predict(self, x):
        return self.h.predict(self.drop_prot(self.dataset, x))

    def predict_proba(self, x):
        return list(map(lambda x: x[1],self.h.predict_proba(self.drop_prot(self.dataset, x))))

    def drop_prot(self, dataset, x):
        return _drop_protected(dataset, np.array(x)) if self.exclude_protected else x

    def accuracy(self, dataset):
        return self.h.score(self.drop_prot(dataset, dataset.features), dataset.labels.ravel())
