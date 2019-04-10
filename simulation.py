import numpy as np
import matplotlib.pyplot as plt

from transformer import AgentTransformer

class Simulation(object):
    def __init__(self, dataset, AgentCl, LearnerCl, cost_distribution):
        self.dataset = dataset
        self.cost_distribution = cost_distribution
        self.LearnerCl = LearnerCl
        self.AgentCl = AgentCl


    def start_simulation(self, include_protected=True):

        # learner moves
        self.learner = self.LearnerCl()
        h = self.learner.fit(self.dataset.features,self.dataset.labels.ravel())

        self.Y_predicted = h(self.dataset.features)

        # agents move
        at = AgentTransformer(self.AgentCl, h, self.cost_distribution)
        dataset_ = at.transform(self.dataset)

        print("Accuracy (h) pre",self.learner.accuracy(self.dataset.features,self.dataset.labels.ravel()))

        # update changed features
        self.Y_new_predicted = h(dataset_.features)

        print("Accuracy (h) post",self.learner.accuracy(dataset_.features, dataset_.labels.ravel()))

        # fit data again, see if accuracy changes
        learner_ = self.LearnerCl()
        learner_.fit(dataset_.features, dataset_.labels.ravel())
        print("Accuracy (h*) post",learner_.accuracy(dataset_.features, dataset_.labels.ravel()))
        print(sum(dataset_.labels.ravel())," <- ", sum(self.dataset.labels.ravel()))

        self.dataset_new = dataset_

        self.dataset_df = self.dataset.convert_to_dataframe(de_dummy_code=True)[0]
        self.dataset_new_df = self.dataset_new.convert_to_dataframe(de_dummy_code=True)[0]


    def show_plots(self):
        plt.show()

    def plot_mutable_features(self):
        disc_and_mutable = self.dataset._discrete_and_mutable()
        plot_ft = lambda ft: self.plot_x(ft, self.dataset_df[ft], self.dataset_new_df[ft])
        for ft in disc_and_mutable:
            plot_ft(ft)

    def plot_group_y(self, time='pre'):
        return
        title = ''
        if time == 'pre':
            X = self.dataset.features
            y = self.Y_predicted
            title = 'Pre'
        elif time == 'post':
            X = self.dataset_new.features
            y = self.Y_new_predicted
            title = 'Post'

        y_domain = set(y)

        grp_feat_name = self.feat_desc.group_feature()
        grps = self.feat_desc.get_groups()
        no_grps = len(grps)

        fig, axs = plt.subplots(1, no_grps)

        count_grp_y = lambda y_, g_: len(list(filter(lambda tpl: tpl[1] == y_ and tpl[0] == g_, zip(X[grp_feat_name],y))))



        for i,g in enumerate(grps):
            axs[i].set_title("Group " + str(g) + ", " + title)
            axs[i].pie(list(map(lambda y_: count_grp_y(y_,g), y_domain)), labels=y_domain, autopct='%1.1f%%', shadow=True)


    def plot_x(self, text, data, data_new):
        plt.figure(text)
        n, bins, patches = plt.hist(x=[sorted(data), sorted(data_new)], label=['pre', 'post'], bins=10,
                                    alpha=0.7, rwidth=0.85)
#        plt.xlim(left=min(min(data_new),min(data)), right=max(max(data_new),max(data)))
        plt.legend(prop={'size': 10})
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(text)
        # plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = (n[0] + n[1]).max()
        # Set a clean upper y-axis limit.
        plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

        plt.show(block=False)



