import numpy as np

class Perceptron:
    def __init__(self, n_features, lr=0.1, epochs=110):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        if X.size == 0 or y.size == 0:
            raise ValueError("Les données d'entraînement ne peuvent pas être vides.")

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self.activation(linear_output)
                update = self.lr * (target - y_pred)
                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)
