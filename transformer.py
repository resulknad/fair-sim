import matplotlib.pyplot as plt
import time
import traceback

from aif360.algorithms import Transformer
from sklearn.neighbors import KNeighborsClassifier
from scipy.optimize import approx_fprime
import numpy as np
import pandas as pd
import asyncio
import uuid

class AgentTransformer(Transformer):
# this class is not thread safe
    def __init__(self, agent_class, h, cost_distribution, scaler, no_neighbors=51, collect_incentive_data=False, avg_out_incentive=1, cost_distribution_dep=None, use_rank=True):

        self.avg_out_incentive = avg_out_incentive
        self.use_rank = True
        self.agent_class = agent_class
        self.h = h
        self.cost_distribution = cost_distribution
        self.no_neighbors = no_neighbors
        self.collect_incentive_data = collect_incentive_data
        self.cost_distribution_dep = cost_distribution_dep
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
            future_copy = self.async_future_list.copy()
            self.async_future_list = []
            #print(self.h)
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

        # fixed cost may be same for all instances
        if self.cost_distribution_dep is None:
            cost = self.cost_distribution(len(dataset.features))
        else:
        # or different depending on features (like group)
            cost = np.array(list(map(self.cost_distribution_dep, dataset.features)))

        dataset_ = dataset.copy(deepcopy=True)
        features_ = []
        labels_ = []
        changed_indices = []

        i=0
        grp0 = []
        grp1 = []

        task_list = []
        self.async_no_wait = len(dataset.features) #* self.avg_out_incentive
        #print("need to wait on ", self.async_no_wait)
        for x,y,c in zip(dataset.features, dataset.labels, cost):
            task_list.append(asyncio.create_task(self._optimal_x(dataset, x, y, c)))

        for task,x,y,c in zip(task_list,dataset.features, dataset.labels, cost):
            await task
            incentive, x_vec = task.result()
            #print(incentive,x_vec)

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
            #print(features_)
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
        nbrs = KNeighborsClassifier(n_neighbors=self.no_neighbors, weights='distance').fit(dataset.features[:,unprotected_feature_indices], dataset.labels.ravel())
        labels = nbrs.predict(X_changed[:,unprotected_feature_indices])
        Y[changed_indices] = list(map(lambda x: [x],labels))


#        assert(Y.sum() >= dataset.labels.sum())
        # update labels (ground truth)
        dataset_.features = X
        dataset_.labels = np.array(Y.tolist())
        #print(dataset_.labels.sum(), " before: ", dataset.labels.sum())
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

    #async def _optimal_x(self, dataset, x, y, cost):
        #scipy.optimize.approx_fprime

    async def _optimal_x(self, dataset, x, y, cost):
        uid = uuid.uuid4().hex
        # x0
        a = self.agent_class(self.h_async, dataset, cost, [x], y)

        # max tracking
        opt_incentive = -1
        opt_x = x
        x_mod_vec = x.copy()

        # iterate over all discrete permutations
        ft_names, permutations = dataset.discrete_permutations()
        ft_changed_ind = list(map(lambda y: dataset.feature_names.index(y), ft_names))
        #incentives = []
        j = 0
        for p in permutations:
            #print(j)
            j+=1
            #p_obj = {k:v for k,v in zip(ft_names,p)}

            # modified x
            #x_ = {**x_obj, **p_obj}
            #x_mod_vec = dataset.obj_to_vector(x_)
            x_mod_vec[ft_changed_ind] = p

            incentive = []
            # update opt if better


            for k in range(self.avg_out_incentive):
                incentive.append(a.incentive([x_mod_vec]))
            incentive = await asyncio.gather(*incentive, return_exceptions=True)


            for i in incentive:
                if hasattr(i, '__traceback__'):
                    traceback.print_tb(i.__traceback__)
            incentive = np.mean(incentive)
            #cost = await a.cost([(x_mod_vec)])
            #benefit = await a.benefit([(x_mod_vec)])
        #    incentives.append([x_['x'], incentive, cost, benefit])
            if incentive > opt_incentive:
                opt_incentive, opt_x = incentive, x_mod_vec.copy()

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

        #print(x,"new opt",opt_x,"with inc", opt_incentive)
        return opt_incentive, opt_x

            # TODO: continious optimization

