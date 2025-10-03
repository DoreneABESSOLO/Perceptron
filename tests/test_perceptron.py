import numpy as np
import pytest
import logging
from perceptron import Perceptron

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def test_initialisation():
    logger.info("Test: Initialisation du perceptron")
    p = Perceptron(n_features=2, lr=0.1)
    assert p.weights.shape == (2,)
    assert isinstance(p.bias, float)

def test_predict_linearly_separable():
    logger.info("Test: Fonction AND logique")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,-1,-1,1])
    p = Perceptron(n_features=2, lr=0.1, epochs=10)
    p.fit(X, y)
    preds = p.predict(X)
    assert (preds == y).all()

def test_learning_rate_effect():
    logger.info("Test: Effet du learning rate")
    X = np.array([[0],[1]])
    y = np.array([-1,1])
    p1 = Perceptron(n_features=1, lr=0.01, epochs=10)
    p2 = Perceptron(n_features=1, lr=1.0, epochs=10)
    p1.fit(X, y)
    p2.fit(X, y)
    assert p1.weights[0] != p2.weights[0]

def test_classification_accuracy():
    logger.info("Test: Dataset sklearn")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2,
                               n_informative=2, n_redundant=0, random_state=42)
    y = np.where(y==0, -1, 1)
    p = Perceptron(n_features=2, lr=0.1, epochs=20)
    p.fit(X, y)
    acc = (p.predict(X) == y).mean()
    assert acc > 0.8

def test_invalid_inputs():
    logger.info("Test: Données invalides")
    p = Perceptron(n_features=2, lr=0.1)
    with pytest.raises(ValueError):
        p.fit(np.array([]), np.array([]))

def test_non_separable_data():
    logger.info("Test: Cas XOR non séparable")
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,1,1,-1])
    p = Perceptron(n_features=2, lr=0.1, epochs=50)
    p.fit(X, y)
    acc = (p.predict(X) == y).mean()
    assert acc < 1.0
