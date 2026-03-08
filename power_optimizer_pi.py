import numpy as np
from sklearn.linear_model import SGDRegressor

class PowerOptimizer:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.model = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=max_iter)
        self.X = None
        self.y = None

    def partial_fit(self, X, y):
        if self.X is None:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.concatenate((self.y, y))
        self.model.partial_fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def get_parameters(self):
        return self.model.coef_, self.model.intercept_