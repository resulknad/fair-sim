import numpy as np
import pandas as pd

from .logisticlearner import LogisticLearner
from .metafairlearner import MetaFairLearner
from .rejectoptionslearner import RejectOptionsLogisticLearner
from .gaussiannblearner import GaussianNBLearner
from .reweighinglearner import ReweighingLogisticLearner
from .statisticalparitybooster import StatisticalParityLogisticLearner
from .statisticalparityflipper import StatisticalParityFlipperLogisticLearner
from .fairlearnlearner import FairLearnLearner
from .eqoddslearner import EqOddsPostprocessingLogisticLearner

from scipy.optimize import minimize


#from sklearn.linear_model import LinearRegression



#from sklearn import svm
from sklearn.model_selection import cross_val_score

#from aif360.algorithms.inprocessing import PrejudiceRemover



#from scipy.optimize import minimize
#import matplotlib.pyplot as plt
#from itertools import product

#from scipy import optimize
#from utils import _df_selection

from aif360.algorithms.inprocessing.meta_fair_classifier import MetaFairClassifier


