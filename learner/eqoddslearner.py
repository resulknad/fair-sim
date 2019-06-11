from aif360.algorithms.postprocessing import EqOddsPostprocessing
class EqOddsPostprocessingLogisticLearner(object):
    def __init__(self, privileged_groups, unprivileged_groups):
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups

    def fit(self, dataset):
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(dataset.features, dataset.labels.ravel())

        dataset_p = dataset.copy()
        dataset_p.scores = np.array(list(map(lambda x: x[1],reg.predict_proba(dataset.features))))
        #dataset_p.labels = np.array(list(map(lambda x: [x], reg.predict(dataset.features))))

        eqodds = EqOddsPostprocessing(unprivileged_groups=self.unprivileged_group, privileged_groups=self.privileged_group)
        eqodds.fit(dataset, dataset_p)

        def h(x, single=True):
            # library does not support datasets with single instances
            if single:
                raise NotImplementedError

            # add dummy labels as we're going to predict them anyway...
            x_with_labels = np.hstack((x, list(map(lambda x: [x], reg.predict(x)))))
            scores = list(map(lambda x:x[1],reg.predict_proba(x)))
            if single:
                assert(len(x_with_labels) == 1)
                x_with_labels = np.repeat(x_with_labels, 100, axis=0)
                scores = np.repeat(scores, 100, axis=0)
            dataset_ = Simulation.dataset_from_matrix(x_with_labels, dataset)
            dataset_.scores = np.array(scores)
            labels_pre = dataset_.labels

            dataset_ = eqodds.predict(dataset_)

            #if not (labels_pre == dataset_.labels).all():
            #    print("fav:",dataset_.favorable_label)
            #    print("labels did change after eqodds.", labels_pre[0][0],"to", dataset_.labels,"for group",x_with_labels[0][1])
            #else:
            #    print("did not change", len(x_with_labels))
            return dataset_.labels.ravel()

        self.h = h
        return h


    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)

