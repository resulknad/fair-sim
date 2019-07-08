:mod:`learner`
========================
All the learners must implement:

- fit
- predict
- predict_proba
- accuracy

.. autoclass:: learner.CalibratedLogisticLearner
  :members:
  :private-members:

.. autoclass:: learner.LogisticLearner
  :members:
  :private-members:

.. autoclass:: learner.MetaFairLearner
  :members:
  :private-members:

.. autoclass:: learner.RejectOptionsLogisticLearner
  :members:
  :private-members:

.. autoclass:: learner.GaussianNBLearner
  :members:
  :private-members:

.. autoclass:: learner.ReweighingLogisticLearner
  :members:
  :private-members:

.. autoclass:: learner.FairLearnLearner
  :members:
  :private-members:

.. automodule:: learner.utils
  :members:
  :private-members:

