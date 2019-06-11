
class MetaFairLearner(object):
    def __init__(self, privileged_group, unprivileged_group):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group

    def fit(self, dataset):
        # sanity checks
        assert(len(self.privileged_group)==1)
        assert((self.privileged_group[0].keys()==self.unprivileged_group[0].keys()))
        class_attr = list(self.privileged_group[0].keys())[0]
        self.dataset = dataset

        self.mfc = MetaFairClassifier(sensitive_attr=class_attr)
        self.mfc.fit(dataset)

    def predict_proba(self, x):
        x_with_labels = np.hstack((x, [[0]] * len(x)))
        dataset_ = Simulation.dataset_from_matrix(x_with_labels, self.dataset)

        scores = self.mfc.predict(dataset_).scores
        print(scores)
        return scores

    def predict(self, x):
        x_with_labels = np.hstack((x, [[0]] * len(x)))
        dataset_ = Simulation.dataset_from_matrix(x_with_labels, self.dataset)

        return self.mfc.predict(dataset_).labels

    def accuracy(self, dataset):
        return _accuracy(self.predict, dataset)
