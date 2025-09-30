from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class PerceptronModel:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate  
        self.max_iter = max_iter  
        self.weights = None  
        self.bias = None  
# la classe du perceptron a été initialisé avec une vitesse d'apprentissage(le pas qui ajuste le poids)
#le nombre de fois que le modèle parcourt les données, le poids et le biais qui seront initialisés plus tard

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
    #l'entrainement initialise les poids et le biais à zéro
    # le x.shape donne la taille de la matrice d'entrée num_samples exemples et les caractéristiques num_features
        
        for _ in range(self.max_iter):
            for idx, sample in enumerate(X):
                linear_output = np.dot(sample, self.weights) + self.bias
                y_predicted = self._activation_function(linear_output)
        
        
                update = self.learning_rate * (y[idx] - y_predicted)  
                self.weights += update * sample
                self.bias += update

        #ici pour chque échantillon, on calcule la sortie linéaire z=X.W+b et
        # et on applique la fonction d'activation au seuil de 0 puis on calcule l'erreur et on met à jour le poids et le biais

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation_function(linear_output)
        return y_predicted
    
    # pour la prédiction,on multiplie les features par les poid et on ajoute le biais et on applique la fonction d'activation
    #ceci retourne 1ou 0

    def _activation_function(self, x):
        return np.where(x >= 0, 1, 0)  
    
    #la fonction d'activation reetourne la classe 1 si la valeur est >= à 0 sinon elle renvoie 0

    def evaluate(self, y_true, y_pred):
        accuracy = np.mean(y_true == y_pred)
        report = classification_report(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        return accuracy, report, cm
    
    #pour l'éval du modèle, 