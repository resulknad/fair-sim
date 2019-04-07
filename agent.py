from scipy.optimize import minimize

class AgentContinious(object):
    def __init__(self, cost, x, y, feat_desc, include_protected):
        self.cost_fixed = cost
        self.feat_desc = feat_desc
        self.x = x
        self.y = y
        self.include_protected = include_protected

    def benefit(self, h, x_new):
        return h(x_new) - h(self.x)

    def cost(self, x_new):
        mutable_features = self.feat_desc.mutable_features()

        ft_cost = sum([ft['cost_fn'](x_new[ft['name']], self.x[ft['name']]) for ft in mutable_features])
        return self.cost_fixed + ft_cost

    def incentive(self, h, x_new):
        return self.benefit(h,x_new) - self.cost(x_new)

    def act(self, h):
        mutable_features = self.feat_desc.mutable_features()

        h_mod = lambda x: h([self.feat_desc.vector_representation(x, self.include_protected)])

        utility = lambda x: -1 * self.incentive(h_mod, x)

        x0 = [self.x[ft['name']] for ft in mutable_features]

        mutable_arr_to_full_feature_dict = lambda x_: {**self.x, **{v['name']: x_[i] for i,v in enumerate(mutable_features)}}

        # minimize requires x = [ ... ], utility_mutable: [ ... ] -> utility({ft: val, ...})
        utility_mutable = lambda x_: utility(mutable_arr_to_full_feature_dict(x_))

        # limit x
        # cons = ({'type': 'ineq', 'fun': lambda x: 20-np.abs(self.x-x)}) , constraints=cons
        res = minimize(utility_mutable, x0)

        if utility_mutable(res.x) < 0:
            self.x = mutable_arr_to_full_feature_dict(res.x)



class AgentDiscrete(object):
    def __init__(self, cost, x, y, feat_desc, include_protected):
        self.cost_fixed = cost
        self.feat_desc = feat_desc
        self.x = x
        self.y = y
        self.include_protected = include_protected

    def benefit(self, h, x_new):
        return h(x_new) - h(self.x)

    def cost(self, x_new):
        return self.cost_fixed

    def incentive(self, h, x_new):
        return self.benefit(h,x_new) - self.cost(x_new)

    def act(self, h):
        h_mod = lambda x: h([self.feat_desc.vector_representation(x, self.include_protected)])

        utility = lambda x: self.incentive(h_mod, x)

        names, permutations = self.feat_desc.iter_permutations()
        max_sofar = (self.x, 0)
        for p in permutations:
            x_new = dict(self.x)
            x_new.update({k: v for k,v in zip(names, p)})
            if utility(x_new) > max_sofar[1]:
                max_sofar = (x_new, utility(x_new))

        if max_sofar[1] > 0:
            print("Change ")
            print(self.x)
            print(" to ")
            print(max_sofar[0])
            print(max_sofar[1])
            self.x = max_sofar[0]
            self.y = h_mod(self.x)
        else:
            print("no change")


