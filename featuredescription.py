import numpy as np
import itertools

class FeatureDescription:
    def __init__(self, X, y):
        self.features = []
        self.X = X
        self.y = y

    def shape(self):
        return (len(self.features), len(self.y))

    def add_descr(self, name, mutable=True, domain={}, cost_fn=(lambda x:x), protected=False, group=False):
        self.features.append({'name': name, 'mutable': mutable, 'domain': domain, 'cost_fn': cost_fn, 'protected': protected, 'group': group})

    def iter_permutations(self):
        mutable_features = self.mutable_features()
        feature_names = tuple(map(lambda x: x['name'], mutable_features))
        domains = [x['domain'] for x in mutable_features]
        crossproduct_iter = itertools.product(*domains)

        return (feature_names,crossproduct_iter)

    def iter_X(self):
        return [{v: self.X[v][i] for v in self.X.keys()} for i in range(self.shape()[1])]

    def unprotected_features(self):
        return list(filter(lambda ft: not ft['protected'], self.features))

    def matrix_representation(self, X=[], include_protected=True):
        if len(X) == 0:
            X = self.X

        features = self.features if include_protected else self.unprotected_features()
        return np.matrix([X[ft['name']] for ft in features]).transpose()

    def vector_representation(self, x, include_protected=True):
        features = self.features if include_protected else self.unprotected_features()
        return [x[ft['name']] for ft in features]

    def combine_named_representations(self, data):
        return {k['name']: [data[i][k['name']] for i in range(self.shape()[1])] for k in self.features}

    def mutable_features(self):
        return list(filter(lambda x: x['mutable'], self.features))

    def group_feature(self):
        group = list(filter(lambda x: x['group'], self.features))
        assert(len(group) == 1)
        return group[0]['name']

    def get_groups(self):
        return set(self.X[self.group_feature()])
