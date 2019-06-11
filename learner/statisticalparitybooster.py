
class StatisticalParityLogisticLearner(object):
    def __init__(self, privileged_group, unprivileged_group, eps, exclude_protected=False):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.eps = eps
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        def drop_prot(x):
            return _drop_protected(dataset, x) if self.exclude_protected else x
        reg = LogisticRegression(solver='liblinear',max_iter=1000000000, C=1000000000000000000000.0).fit(drop_prot(dataset.features), dataset.labels.ravel())
        self.h = reg

        assert(len(self.privileged_group)==1)
        assert(len(dataset.label_names) == 1)

        #df = dataset.convert_to_dataframe()[0].drop(columns=dataset.label_names)

        dataset.features = np.array(list(map(np.array, dataset.features)))
        df = pd.DataFrame(data=dataset.features, columns=dataset.feature_names)
        # priv count
        #print(self.privileged_group[0])
        n_p = len(_df_selection(df, self.privileged_group[0]).values)

        # not priv count
        n_np = len(_df_selection(df, self.unprivileged_group[0]).values)

        def stat_parity_diff(boost):
            data = _df_selection(df, self.privileged_group[0])
            probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
            n_p_y = (np.array(probs)>0.5).sum()

            data = _df_selection(df, self.unprivileged_group[0])
            assert(len(data.values)<=n_np)
            probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
            n_np_y = (np.array(np.add(probs,[boost]*len(probs)))>0.5).sum()


            stat_par_diff = n_np_y/n_np -  n_p_y/n_p
            if boost == 0:
                return stat_par_diff
            else:
                return stat_par_diff# - stat_parity_diff(0)/2.
            #print("Boost:",boost,n_np_y,"of", n_np, n_p_y,"of",n_p,stat_par_diff)
            return stat_par_diff
        try:
            boost = optimize.bisect(stat_parity_diff, 0, 1, xtol=self.eps, disp=True)
        except ValueError: #
            print("couldnt find appropriate boost, dont boost")
            boost = 0
        #boost = 0.
        print("Boost:",boost)

        group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
        grp_i = dataset.feature_names.index(group_ft)
        def decision_fn(x,single=True):
            ys = []
            probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(x))))
            for x,p in zip(x,probs):
                p_ = p + boost if x[grp_i] == unpriv_val else p
                if p_ > 0.5:
                    ys.append(1)
                else:
                    ys.append(0)
            return ys[0] if single else ys

        self.h = decision_fn
        return decision_fn #lambda x: 1 if np.add(reg.predict_proba(x)[1],x[grp_i]*boost > 0.5 else 0

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)
