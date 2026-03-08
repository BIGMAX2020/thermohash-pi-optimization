import numpy as np
from sklearn.linear_model import SGDRegressor

class PowerOptimizer:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.model = SGDRegressor(learning_rate='constant', eta0=learning_rate, max_iter=max_iter)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

# Example Usage
if __name__ == '__main__':
    # Sample data
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([1, 2, 3])
    optimizer = PowerOptimizer(learning_rate=0.01)
    optimizer.fit(X, y)
    predictions = optimizer.predict(X)
    print(predictions)