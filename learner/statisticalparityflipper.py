class StatisticalParityFlipperLogisticLearner(object):
    def __init__(self, privileged_group, unprivileged_group, exclude_protected=False):
        self.privileged_group = privileged_group
        self.unprivileged_group = unprivileged_group
        self.ratio = 0.5
        self.exclude_protected = exclude_protected

    def fit(self, dataset):
        def drop_prot(x):
            return _drop_protected(dataset, x) if self.exclude_protected else x

        reg = LogisticRegression(solver='lbfgs',max_iter=1000000000, C=1000000000000000000000.0).fit(drop_prot(dataset.features), dataset.labels.ravel())
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

        data = _df_selection(df, self.privileged_group[0])
        probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
        n_p_y = (np.array(probs)>0.5).sum()

        data = _df_selection(df, self.unprivileged_group[0])
        assert(len(data.values)<=n_np)
        probs = list(map(lambda x: x[1], reg.predict_proba(drop_prot(data.values))))
        n_np_y = (np.array(probs)>0.5).sum()

        stat_par_diff = n_p_y/n_p - n_np_y/n_np

        # unprivileged should be unprivileged...

        group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
        grp_i = dataset.feature_names.index(group_ft)

        def decision_fn(x,single=True):
            x = np.array(x)
            y_pred = reg.predict(drop_prot(x))


            def flip(grp_val, flip_from, flip_to, fraction):
                # select all from group in x
                grp_ind = x[:,grp_i] == grp_val
                # find indices we predicted flip_from
                label_ind = (y_pred == flip_from)
                # flippable indices
                flippable = (grp_ind & label_ind).nonzero()[0]
                # shuffle
                flippable = np.random.permutation(flippable)
                # truncate
                truncated_size = int(round(abs(stat_par_diff)*grp_ind.sum()*fraction))
                #print(grp_val,"stat par diff is", stat_par_diff, " so we flip ",truncated_size)
                flippable = flippable[:truncated_size]

                # flip
                y_pred[flippable] = [flip_to]* len(flippable)

            if stat_par_diff > 0:
                flip(unpriv_val, 0, 1, 1)
                #flip(1-unpriv_val, 1, 0, .5)
            else:
                flip(unpriv_val, 1, 0, 1)
                #flip(1-unpriv_val, 0, 1, .5)

            #print("flipped ", len(flippable))


            return y_pred.tolist()

        #if stat_par_diff > 0:
            #print("flipping up from 0 to 1 with pr", stat_par_diff/2., " for p from 1 to 0 with same pr")
        #else:
            #print("flipping up from 1 to 0 with pr", stat_par_diff/2., " for p from 0 to 1 with same pr")


        self.h = decision_fn
        return decision_fn #lambda x: 1 if np.add(reg.predict_proba(x)[1],x[grp_i]*boost > 0.5 else 0

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)
