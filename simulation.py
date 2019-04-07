import numpy as np
import matplotlib.pyplot as plt

class Simulation(object):
    def __init__(self, feat_desc, AgentCl, LearnerCl, cost_distribution):
        self.feat_desc = feat_desc
        self.X = feat_desc.X
        self.Y = feat_desc.y
        self.cost_distribution = cost_distribution
        self.LearnerCl = LearnerCl
        self.AgentCl = AgentCl


    def start_simulation(self, include_protected=True):
        # draw cost
        self.X_cost = np.abs(self.cost_distribution(self.feat_desc.shape()[1]))

        # learner moves
        self.learner = self.LearnerCl()
        h = self.learner.fit(self.feat_desc.matrix_representation(include_protected=include_protected),self.Y)

        # agents learn about cost
        self.agents = [self.AgentCl(cost, x, y, self.feat_desc, include_protected) for (cost, x, y) in zip(self.X_cost, self.feat_desc.iter_X(), self.Y)]
        for a in self.agents[:]:
            a.act(h)

        print("Accuracy (h) pre",self.learner.accuracy(self.feat_desc.matrix_representation(include_protected=include_protected),self.Y))

        # update changed features
        X_new = (self.feat_desc.combine_named_representations([a.x for a in self.agents]))
        y_new = ([a.y for a in self.agents])
        self.X_new = X_new
        self.y_new = y_new

        print("Accuracy (h) post",self.learner.accuracy(self.feat_desc.matrix_representation(X_new, include_protected=include_protected),y_new))

        # fit data again, see if accuracy changes
        learner_ = self.LearnerCl()
        learner_.fit(self.feat_desc.matrix_representation(X_new,include_protected=include_protected),y_new)
        print("Accuracy (h*) post",learner_.accuracy(self.feat_desc.matrix_representation(X_new, include_protected=include_protected),y_new))

        print(sum(y_new)," <- ", sum(self.Y))


        #plot h
        #xvals = np.linspace(-100,100,1000) #100 points from 0 to 6 in ndarray
        #yvals = list(map(lambda x: h([x]), xvals)) #evaluate f for each point in xvals
        #plt.figure("h")
        #plt.plot(xvals, yvals)


    def show_plots(self):
        plt.show()

    def plot_mutable_features(self):
        plot_ft = lambda ft: self.plot_x(ft, self.X[ft], self.X_new[ft])
        for ft in map(lambda x: x['name'], self.feat_desc.mutable_features()):
            plot_ft(ft)

    def plot_group_y(self, time='pre'):
        title = ''
        if time == 'pre':
            X = self.X
            y = self.Y
            title = 'Pre'
        elif time == 'post':
            X = self.X_new
            y = self.y_new
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
        n, bins, patches = plt.hist(x=[data, data_new], label=['pre', 'post'], bins=10,
                                    alpha=0.7, rwidth=0.85)
        plt.xlim(left=min(min(data_new),min(data)), right=max(max(data_new),max(data)))
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
        print(np.average(data_new),"<-",np.average(data))



