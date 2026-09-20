import math
from typing import List

class LinearRegression:
    """
    Simple Multivariate Linear Regression using Batch Gradient Descent.
    Implemented in pure Python without external dependencies.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[float]):
        n_samples = len(X)
        if n_samples == 0:
            raise ValueError("Input dataset X cannot be empty.")
        n_features = len(X[0])

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            # Predictions
            y_pred = []
            for row in X:
                val = self.bias + sum(w * x for w, x in zip(self.weights, row))
                y_pred.append(val)

            # Gradients
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                err = y_pred[i] - y[i]
                db += err
                for j in range(n_features):
                    dw[j] += err * X[i][j]

            # Update parameters
            self.bias -= (self.lr * db) / n_samples
            for j in range(n_features):
                self.weights[j] -= (self.lr * dw[j]) / n_samples

    def predict(self, X: List[List[float]]) -> List[float]:
        predictions = []
        for row in X:
            val = self.bias + sum(w * x for w, x in zip(self.weights, row))
            predictions.append(val)
        return predictions
