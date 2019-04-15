
from aif360.algorithms import Transformer
import numpy as np



class AgentTransformer(Transformer):
    def __init__(self, agent_class, h, cost_distribution):
        self.agent_class = agent_class
        self.h = h
        self.cost_distribution = cost_distribution
        super(AgentTransformer, self).__init__(
            agent_class=agent_class,
            h=h,
            cost_distribution=cost_distribution)

    def transform(self, dataset):
        cost = self.cost_distribution(len(dataset.features))

        dataset_ = dataset.copy()
        features_ = []
        labels_ = []
        for x,y,c in zip(dataset.features, dataset.labels, cost):
            incentive, x_ = self._optimal_x(dataset, x, y, c)
            if incentive > 0:
                x_vec = dataset.obj_to_vector(x_)
                features_.append(x_vec)
                labels_.append([self.h([x_vec])])

                assert(not (x_vec == x).all())
                assert(self.h([x])!=self.h([x_vec]))
            else:
                features_.append(x)
                labels_.append(y)

        dataset_.features = features_
        dataset_.labels = np.array(labels_)
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

        for p in permutations:
            p_obj = {k:v for k,v in zip(ft_names,p)}

            # modified x
            x_ = {**x_obj, **p_obj}


            # update opt if better
            incentive = a.incentive([dataset.obj_to_vector(x_)])
            if incentive > opt_incentive:
                opt_incentive, opt_x = incentive, x_

        return opt_incentive, opt_x

            # TODO: continious optimization

