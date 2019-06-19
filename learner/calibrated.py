import numpy as np
from utils import dataset_from_matrix
from .utils import _accuracy, _drop_protected
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit

from sklearn.model_selection import KFold

class CalibratedLogisticLearner(object):
    threshold = 0.5
    def __init__(self, privileged_groups, unprivileged_groups):
        self.privileged_group = privileged_groups
        self.unprivileged_group = unprivileged_groups

    def fit(self, dataset):
        self.dataset = dataset
        self.coefs = []

        # split in train and test (test used for calibration)
        group_ft, unpriv_val = list(self.unprivileged_group[0].items())[0]
        _, priv_val = list(self.privileged_group[0].items())[0]
        grp_ind = dataset.feature_names.index(group_ft)

        X = dataset.features
        y = dataset.labels.ravel()

        priv_ind = X[:,grp_ind] != unpriv_val
        unpriv_ind = X[:,grp_ind] == unpriv_val

        p_split = KFold(n_splits=3).split(X[priv_ind], y[priv_ind])
        up_split = KFold(n_splits=3).split(X[unpriv_ind], y[unpriv_ind])

        classifiers_up = []
        classifiers_p = []
        for up, p in zip(up_split, p_split):
            up_train, up_test = up
            p_train, p_test = p
            X_p_train = X[priv_ind][p_train]
            y_p_train = y[priv_ind][p_train]
            X_up_train = X[unpriv_ind][up_train]
            y_up_train = y[unpriv_ind][up_train]

            X_p_test = X[priv_ind][p_test]
            y_p_test = y[priv_ind][p_test]
            X_up_test = X[unpriv_ind][up_test]
            y_up_test = y[unpriv_ind][up_test]

            X_train = np.vstack((X_p_train, X_up_train))
            y_train = np.hstack((y_p_train, y_up_train))

            reg = LogisticRegression(solver='liblinear',max_iter=1000000000).fit(_drop_protected(self.dataset, X_train), y_train)

            self.coefs.append((sorted(list(zip(dataset.feature_names,reg.coef_[0])),key=lambda x: -abs(x[1]))))

            cal_p = CalibratedClassifierCV(reg, cv='prefit')
            cal_p.fit(_drop_protected(self.dataset, X_p_test), y_p_test)
            classifiers_p.append(cal_p)
            print(cal_p.get_params())

            cal_up = CalibratedClassifierCV(reg, cv='prefit')
            cal_up.fit(_drop_protected(self.dataset, X_up_test), y_up_test)
            classifiers_up.append(cal_up)
            print(cal_up.get_params())

        def h_pr(x):
            scores = np.zeros(len(x))

            # drop protected attributes
            x_wo_prot = _drop_protected(self.dataset, x)

            priv_ind = x[:,grp_ind] != unpriv_val
            unpriv_ind = x[:,grp_ind] == unpriv_val

            for cal_p in classifiers_p:
                scores[priv_ind] += cal_p.predict_proba(x_wo_prot[priv_ind])[:,1]
            scores[priv_ind] /= len(classifiers_p)

            for cal_p in classifiers_up:
                scores[unpriv_ind] += cal_up.predict_proba(x_wo_prot[unpriv_ind])[:,1]
            scores[unpriv_ind] /= len(classifiers_up)

            return np.array(scores)

        def h(x):
            return (h_pr(x)>0.5).astype(int)

        self.h_pr = h_pr
        self.h = h

    def bak(self):
        if False:
            from sklearn.calibration import calibration_curve

            p_y, p_x = calibration_curve(y_p_test, reg.predict_proba(_drop_protected(self.dataset, X_p_test))[:,1], n_bins=10)
            up_y, up_x = calibration_curve(y_up_test, reg.predict_proba(_drop_protected(self.dataset, X_up_test))[:,1], n_bins=10)

            import matplotlib.pyplot as plt
            import matplotlib.lines as mlines
            import matplotlib.transforms as mtransforms


            fig, ax = plt.subplots()
            # only these two lines are calibration curves
            plt.plot(p_x,p_y, marker='o', linewidth=1, label='p')
            plt.plot(up_x, up_y, marker='o', linewidth=1, label='up')

            # reference line, legends, and axis labels
            line = mlines.Line2D([0, 1], [0, 1], color='black')
            transform = ax.transAxes
            line.set_transform(transform)
            ax.add_line(line)
            fig.suptitle('Calibration plot for Titanic data')
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('True probability in each bin')
            plt.legend()
            #plt.show()

    def predict_proba(self, x):
        return self.h_pr(x)

    def predict(self, x):
        return self.h(x)

    def accuracy(self, dataset):
        return _accuracy(self.h, dataset)
