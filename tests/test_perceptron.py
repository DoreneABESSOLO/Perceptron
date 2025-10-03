"""
Fichier : test_perceptron.py
Description : Tests unitaires et fonctionnels pour valider l'implémentation
              du Perceptron (perceptron.py).

Chaque test est documenté et utilise des logs (via le module logging)
pour mieux comprendre le déroulement des vérifications.
"""

import numpy as np
import pytest
import logging
from src import Perceptron  # La classe que tu as implémentée dans perceptron.py

# Configuration du logger pour pytest
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_initialisation():
    """
    Vérifie que le perceptron est bien initialisé.
    """
    logger.info("Test : Initialisation du perceptron")
    p = Perceptron(n_features=2, lr=0.1)
    logger.info(f"Weights initiaux: {p.weights}, Bias initial: {p.bias}")
    assert p.weights.shape == (2,), "Le vecteur de poids doit avoir 2 dimensions"
    assert isinstance(p.bias, float), "Le biais doit être un float"


def test_predict_linearly_separable():
    """
    Vérifie que le perceptron apprend correctement la fonction logique AND.
    """
    logger.info("Test : Apprentissage de la fonction logique AND")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,-1,-1,1])  # AND

    p = Perceptron(n_features=2, lr=0.1, epochs=10)
    p.fit(X, y)
    preds = p.predict(X)
    logger.info(f"Prédictions obtenues: {preds}, attendues: {y}")

    assert (preds == y).all(), "Le perceptron doit apprendre AND parfaitement"


def test_learning_rate_effect():
    """
    Vérifie que le taux d'apprentissage influence bien l'entraînement.
    """
    logger.info("Test : Effet du learning rate")
    X = np.array([[0],[1]])
    y = np.array([-1,1])

    p1 = Perceptron(n_features=1, lr=0.01, epochs=10)
    p2 = Perceptron(n_features=1, lr=1.0, epochs=10)

    p1.fit(X, y)
    p2.fit(X, y)
    logger.info(f"Weights p1: {p1.weights}, Weights p2: {p2.weights}")

    assert p1.weights[0] != p2.weights[0], "Les poids doivent être différents avec des lr différents"


def test_classification_accuracy():
    """
    Vérifie que le perceptron atteint une bonne précision (>80%) sur un dataset jouet.
    """
    logger.info("Test : Précision sur un dataset jouet sklearn")
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100, n_features=2, n_classes=2,
        n_informative=2, n_redundant=0, random_state=42
    )
    y = np.where(y==0, -1, 1)

    p = Perceptron(n_features=2, lr=0.1, epochs=20)
    p.fit(X, y)
    preds = p.predict(X)
    acc = (preds == y).mean()
    logger.info(f"Accuracy obtenue: {acc*100:.2f}%")

    assert acc > 0.8, "La précision doit être supérieure à 80%"


def test_invalid_inputs():
    """
    Vérifie que le perceptron lève une ValueError si on fournit des données vides.
    """
    logger.info("Test : Gestion des entrées invalides")
    p = Perceptron(n_features=2, lr=0.1)
    with pytest.raises(ValueError):
        p.fit(np.array([]), np.array([]))


def test_non_separable_data():
    """
    Vérifie que le perceptron échoue sur des données non séparables (XOR).
    """
    logger.info("Test : Limite du perceptron sur XOR")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,1,1,-1])  # XOR

    p = Perceptron(n_features=2, lr=0.1, epochs=50)
    p.fit(X, y)
    preds = p.predict(X)
    acc = (preds == y).mean()
    logger.info(f"Accuracy sur XOR: {acc*100:.2f}% (attendu < 100%)")

    assert acc < 1.0, "Le perceptron ne peut pas résoudre XOR (non linéairement séparable)"
