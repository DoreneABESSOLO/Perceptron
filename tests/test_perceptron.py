"""
Fichier : test_perceptron.py
Description : Ensemble de tests unitaires et fonctionnels pour valider l'implémentation
              de la classe Perceptron (perceptron.py).

Ces tests utilisent pytest et couvrent :
- L'initialisation correcte de la classe
- La capacité du perceptron à apprendre sur des données séparables
- L'effet du taux d'apprentissage (learning rate)
- La performance sur un jeu de données généré par sklearn
- La gestion des erreurs (inputs invalides)
- Les limites du perceptron (ex : problème XOR non linéaire)
"""

import numpy as np
import pytest
from perceptron import Perceptron  # La classe que tu as implémentée dans perceptron.py


def test_initialisation():
    """
    Vérifie que le perceptron est bien initialisé avec le bon nombre de poids
    et un biais de type float.
    """
    p = Perceptron(n_features=2, lr=0.1)
    assert p.weights.shape == (2,), "Le vecteur de poids doit avoir la bonne dimension"
    assert isinstance(p.bias, float), "Le biais doit être un float"


def test_predict_linearly_separable():
    """
    Vérifie que le perceptron apprend correctement une fonction logique simple ET (AND).
    Cas séparables linéairement : doit réussir à tout classer correctement.
    """
    # Table de vérité ET logique
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,-1,-1,1])  # sorties attendues

    p = Perceptron(n_features=2, lr=0.1, epochs=10)
    p.fit(X, y)
    preds = p.predict(X)

    assert (preds == y).all(), "Le perceptron doit apprendre la fonction AND parfaitement"


def test_learning_rate_effect():
    """
    Vérifie que le taux d'apprentissage (lr) a bien un effet sur l'entraînement.
    Deux perceptrons entraînés avec des lr différents ne devraient pas avoir les mêmes poids.
    """
    X = np.array([[0],[1]])
    y = np.array([-1,1])

    p1 = Perceptron(n_features=1, lr=0.01, epochs=10)
    p2 = Perceptron(n_features=1, lr=1.0, epochs=10)

    p1.fit(X, y)
    p2.fit(X, y)

    assert p1.weights[0] != p2.weights[0], "Les poids doivent être différents avec des lr différents"


def test_classification_accuracy():
    """
    Vérifie la performance du perceptron sur un dataset jouet généré par sklearn.
    Comme il est linéairement séparable, on attend une précision correcte (> 80%).
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100, n_features=2, n_classes=2,
        n_informative=2, n_redundant=0, random_state=42
    )
    y = np.where(y==0, -1, 1)  # Adapter les labels à {-1, 1}

    p = Perceptron(n_features=2, lr=0.1, epochs=20)
    p.fit(X, y)

    acc = (p.predict(X) == y).mean()
    assert acc > 0.8, "Le perceptron doit atteindre au moins 80% de précision sur ce dataset"


def test_invalid_inputs():
    """
    Vérifie que le perceptron lève une erreur si on tente d'entraîner avec des données vides.
    """
    p = Perceptron(n_features=2, lr=0.1)
    with pytest.raises(ValueError):
        p.fit(np.array([]), np.array([]))


def test_non_separable_data():
    """
    Vérifie le comportement du perceptron sur des données non séparables (problème XOR).
    Le perceptron ne doit pas réussir à atteindre 100% de précision (limite connue).
    """
    # Table de vérité XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,1,1,-1])  # sorties attendues

    p = Perceptron(n_features=2, lr=0.1, epochs=50)
    p.fit(X, y)

    acc = (p.predict(X) == y).mean()
    assert acc < 1.0, "Le perceptron ne peut pas apprendre XOR (non linéairement séparable)"
