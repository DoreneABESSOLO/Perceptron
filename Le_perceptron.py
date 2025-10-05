from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

class PerceptronModel:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  
        self.max_iter = max_iter  
        self.weights = None  
        self.bias = None  
        self.losses = []  # pour stocker la loss

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for epoch in range(self.max_iter):
            total_error = 0
            for idx, sample in enumerate(X):
                linear_output = np.dot(sample, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)

                update = self.learning_rate * (y[idx] - y_predicted)  
                self.weights += update * sample
                self.bias += update

                # erreur quadratique
                total_error += (y[idx] - y_predicted) ** 2

            # enregistrer la loss de l'époque
            self.losses.append(total_error)

        print(f"✅ Entraînement terminé ({self.max_iter} époques)")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        probs = self._activation_function(linear_output)
        return np.where(probs >= 0.5, 1, 0)

    def _activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def evaluate(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, report, cm

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Époques")
        plt.ylabel("Loss (erreur cumulée)")
        plt.title("Courbe d'apprentissage du perceptron")
        plt.grid(True)
        plt.show()
