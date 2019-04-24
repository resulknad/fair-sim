import matplotlib.pyplot as plt

from aif360.algorithms import Transformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import asyncio
import uuid

class AgentTransformer(Transformer):
# this class is not thread safe
    def __init__(self, agent_class, h, cost_distribution, scaler, no_neighbors=51, collect_incentive_data=True):
        self.agent_class = agent_class
        self.h = h
        self.cost_distribution = cost_distribution
        self.no_neighbors = no_neighbors
        self.collect_incentive_data = collect_incentive_data
        self.incentives = []



# list to keep track of returned futures from h_async
# if do_async = True
        self.async_future_list = []
# no of elements to wait until execution of h
        self.async_no_wait = 0
# save to xs which get passed to h
        self.async_xs = []


        super(AgentTransformer, self).__init__(
            agent_class=agent_class,
            h=h,
            cost_distribution=cost_distribution)

# returns a future until self.async_no_wait is reached
# then execute self.h with all of the xs
    def h_async(self, x, single=True):
        #print("enqueueing")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.async_future_list.append(future)

        if not single:
            raise NotImplementedError #atm only singles supported

        self.async_xs.append(x[0])

# may now execute h, got enough xs
        if len(self.async_future_list) == self.async_no_wait:
            #print("resolving")
            future_copy = self.async_future_list.copy()
            self.async_future_list = []

            #print(self.async_xs)
            ys = self.h(self.async_xs, single=False)
            self.async_xs = []
            #print(ys)

            for f,y in zip(future_copy, ys):
                #print("resolved", y)
# resolve for consumer
                f.set_result(y)

        return future

    async def _do_simulation(self, dataset):
        # setup incentive data collection
        if self.collect_incentive_data:
            self.incentives = []



        cost = self.cost_distribution(len(dataset.features))

        dataset_ = dataset.copy()
        features_ = []
        labels_ = []
        changed_indices = []

        i=0
        grp0 = []
        grp1 = []

        task_list = []
        self.async_no_wait = len(dataset.features)
        for x,y,c in zip(dataset.features, dataset.labels, cost):
            task_list.append(asyncio.create_task(self._optimal_x(dataset, x, y, c)))

        for task,x,y,c in zip(task_list,dataset.features, dataset.labels, cost):
            await task
            incentive, x_ = task.result()
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

        unprotected_feature_indices = list(map(lambda x: not x in dataset.protected_attribute_names, dataset.feature_names))
        # fit KNN to unchanged (during simulation) datapoints
        nbrs = NearestNeighbors(n_neighbors=self.no_neighbors).fit(X_unchanged[:,unprotected_feature_indices])
        _, indices = nbrs.kneighbors(X_changed[:,unprotected_feature_indices])

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

    def transform(self, dataset):
        task = self._do_simulation(dataset)
        assert(len(self.async_future_list)==0)

        # workaround for jupyter
        try:
            loop = asyncio.get_event_loop()
            dataset_ = loop.run_until_complete(task)
        except RuntimeError:
            dataset_ = asyncio.run(task)
        # create df for incentives
        ft_names_orig = list(map(lambda x: x+"_orig", dataset.feature_names))


        self.incentive_df = pd.DataFrame(data=self.incentives, columns=['uid'] + ft_names_orig + dataset.feature_names + ['incentive'], dtype=float)
        self.incentives=[]

        return dataset_

    async def _optimal_x(self, dataset, x, y, cost):
        uid = uuid.uuid4().hex
        # x0
        x_obj = dataset.vector_to_object(x)
        a = self.agent_class(self.h_async, dataset, cost, [x], y)

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
            x_mod_vec = dataset.obj_to_vector(x_)

            # update opt if better
            incentive = await a.incentive([x_mod_vec])
            cost = await a.cost([(x_mod_vec)])
            benefit = await a.benefit([(x_mod_vec)])
        #    incentives.append([x_['x'], incentive, cost, benefit])
            if incentive > opt_incentive:
                opt_incentive, opt_x = incentive, x_

            if self.collect_incentive_data:
                self.incentives.append(np.hstack(([uid], x,x_mod_vec,[incentive])))
            #print("Option:", incentive, "New",x_['x'], "old",x[0])

        #incentives = sorted(incentives, key=lambda x: x[0])
        #xs = list(map(lambda x: x[0], incentives))
        #iss = list(map(lambda x: x[1], incentives))
        #cost = list(map(lambda x: x[2], incentives))
        #benefit = list(map(lambda x: x[3], incentives))
        #if opt_x['group'] == 1:

            #plt.plot(xs, iss)
            #plt.plot(xs, cost)
            #plt.plot(xs, benefit)
            #plt.show()

        return opt_incentive, opt_x

            # TODO: continious optimization

