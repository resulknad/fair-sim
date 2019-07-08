class GeneralLearner():
    def fit(self,dataset):
        """Fits the classifier on the dataset.

        :param dataset: Some AIF360 dataset
        """
        raise NotImplemented()

    def predict_proba(self, X):
        """Makes predictions for feature matrix x.

        :param X: feature matrix (dimension `num_instances x num_features`)
        :returns: Probability vector
        """
        raise NotImplemented()

    def predict(self, x):
        """Makes predictions for feature matrix x.

        :param X: feature matrix (dimension `num_instances x num_features`)
        :returns: Vector containing the predicted labels
        """
        raise NotImplemented()

    def accuracy(self, dataset):
        """Calculates accuracy of learner for some dataset.

        :param dataset: AIF360 dataset
        :returns: Accuracy ([0,1])
        """
        raise NotImplemented()
