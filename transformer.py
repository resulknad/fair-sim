import matplotlib.pyplot as plt
from aif360.algorithms import Transformer
from sklearn.neighbors import NearestNeighbors
import numpy as np

class AgentTransformer(Transformer):
    def __init__(self, agent_class, h, cost_distribution, scaler, no_neighbors=51):
        self.agent_class = agent_class
        self.h = h
        self.cost_distribution = cost_distribution
        self.no_neighbors = no_neighbors

        super(AgentTransformer, self).__init__(
            agent_class=agent_class,
            h=h,
            cost_distribution=cost_distribution)

    def transform(self, dataset):
        cost = self.cost_distribution(len(dataset.features))

        dataset_ = dataset.copy()
        features_ = []
        labels_ = []
        changed_indices = []

        i=0
        grp0 = []
        grp1 = []
        for x,y,c in zip(dataset.features, dataset.labels, cost):
            incentive, x_ = self._optimal_x(dataset, x, y, c)
            x_vec = dataset.obj_to_vector(x_)

            if incentive > 0 and not (x_vec == x).all():
        #        if x_['group'] == 0:
        #            grp0.append(x_['x'])
        #        elif x_['group'] == 1:
        #            grp1.append(x_['x'])



                features_.append(np.array(x_vec))
                changed_indices.append(i)
                labels_.append([])

                #assert(not (x_vec == x).all())
                #assert(self.h([x])!=self.h([x_vec]))
            else:
        #        if x_['group'] == 0:
        #            grp0.append(x[0])
        #        elif x_['group'] == 1:
        #            grp1.append(x[0])
                features_.append(np.array(x))
                labels_.append(y)
            i+=1

        dataset_.features = features_

        #print("grp0: avg opt x",np.average(np.array(grp0)))
        #print("grp1: avg opt x",np.average(np.array(grp1)))


        X = np.array(features_)
        Y = np.array(labels_)

        # no changes during simulation
        # no need to assign new labels with KNN
        if len(changed_indices) == 0:
            dataset_.features = X
            dataset_.labels = np.array(Y.tolist())
            return dataset_

        unchanged_indices = np.setdiff1d(list(range(len(X))), changed_indices)
        X_changed = X[changed_indices,:]
        Y_changed = Y[changed_indices]

        X_unchanged = X[unchanged_indices,:]
        Y_unchanged = Y[unchanged_indices]
        assert(len(X_changed)==len(changed_indices))
        assert(len(X_unchanged)==len(X)-len(changed_indices))

        # fit KNN to unchanged (during simulation) datapoints
        nbrs = NearestNeighbors(n_neighbors=self.no_neighbors).fit(X_unchanged)
        _, indices = nbrs.kneighbors(X_changed)

        # for all changed datapoints
        for i, x, neighbors in zip(range(len(X_changed)), X_changed,indices):
            # get labels of nearest neighbors
            neighbor_labels = Y_unchanged[neighbors]
            unique, counts = np.unique(neighbor_labels, return_counts=True)

            # set own label to the most common one among neighbors
            _,label = sorted(zip(counts,unique), key=lambda x: x[0], reverse=True)[0]
            Y_changed[i] = label

        Y[changed_indices] = Y_changed

        # update labels (ground truth)
        dataset_.features = X
        dataset_.labels = np.array(Y.tolist())
        return dataset_

    def _optimal_x(self, dataset, x, y, cost):
        # x0
        x_obj = dataset.vector_to_object(x)
        a = self.agent_class(self.h, dataset, cost, [x], y)

        # max tracking
        opt_incentive = -1
        opt_x = x_obj

        # iterate over all discrete permutations
        ft_names, permutations = dataset.discrete_permutations()
        #incentives = []
        for p in permutations:
            p_obj = {k:v for k,v in zip(ft_names,p)}

            # modified x
            x_ = {**x_obj, **p_obj}

            # update opt if better
            incentive = a.incentive([dataset.obj_to_vector(x_)])
            cost = a.cost([dataset.obj_to_vector(x_)])
            benefit = a.benefit([dataset.obj_to_vector(x_)])
        #    incentives.append([x_['x'], incentive, cost, benefit])
            if incentive > opt_incentive:
                opt_incentive, opt_x = incentive, x_

        #incentives = sorted(incentives, key=lambda x: x[0])
        #xs = list(map(lambda x: x[0], incentives))
        #iss = list(map(lambda x: x[1], incentives))
        #cost = list(map(lambda x: x[2], incentives))
        #benefit = list(map(lambda x: x[3], incentives))
        #if opt_x['group'] == 1:
            #print("Best:",y,opt_incentive, "New",opt_x, "old",x)
            #plt.plot(xs, iss)
            #plt.plot(xs, cost)
            #plt.plot(xs, benefit)
            #plt.show()

        return opt_incentive, opt_x

            # TODO: continious optimization

