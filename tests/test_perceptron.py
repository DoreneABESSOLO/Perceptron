"""
Fichier : test_perceptron.py
Description : Tests unitaires du modèle Perceptron avec pytest.
Objectif : Vérifier la robustesse, la précision et le comportement général du modèle.
"""
# === Importations des bibliothèques nécessaires ===
import numpy as np          # Pour la manipulation de tableaux de données
import pytest               # Pour la gestion et l’exécution des tests unitaires
import logging              # Pour enregistrer les logs d'exécution des tests
from perceptron import Perceptron  # Import du modèle à tester

# === Configuration du logger ===
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ==========================================================================================
# 🔹 TEST 1 : Initialisation du modèle
# Vérifie que le perceptron s’initialise correctement avec le bon nombre de poids et un biais.
# ==========================================================================================
def test_initialisation():
    logger.info("Test: Initialisation du perceptron")
    
    # Création d'un perceptron avec 2 features et un taux d'apprentissage de 0.1
    p = Perceptron(n_features=2, lr=0.1)
    
    # Vérification que le vecteur de poids est bien de taille 2
    assert p.weights.shape == (2,)
    
    # Vérification que le biais est bien un nombre flottant
    assert isinstance(p.bias, float)

# ==========================================================================================
# 🔹 TEST 2 : Apprentissage sur un jeu de données linéairement séparable (fonction AND)
# Vérifie que le modèle apprend correctement une logique simple et séparable.
# ==========================================================================================
def test_predict_linearly_separable():
    logger.info("Test: Fonction AND logique")
    
    # Jeu de données pour la fonction logique AND
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,-1,-1,1])  # Sorties attendues
    
    # Entraînement du perceptron
    p = Perceptron(n_features=2, lr=0.1, epochs=10)
    p.fit(X, y)
    
    # Prédiction sur les mêmes données
    preds = p.predict(X)
    
    # Vérifie que toutes les prédictions correspondent à la vérité terrain
    assert (preds == y).all()

# ==========================================================================================
# 🔹 TEST 3 : Effet du learning rate (taux d’apprentissage)
# Vérifie que des taux d’apprentissage différents produisent des poids différents.
# ==========================================================================================
def test_learning_rate_effect():
    logger.info("Test: Effet du learning rate")
    
    # Données simples : une seule feature binaire
    X = np.array([[0],[1]])
    y = np.array([-1,1])
    
    # Deux perceptrons identiques sauf le learning rate
    p1 = Perceptron(n_features=1, lr=0.01, epochs=10)
    p2 = Perceptron(n_features=1, lr=1.0, epochs=10)
    
    # Entraînement
    p1.fit(X, y)
    p2.fit(X, y)
    
    # Vérifie que les poids finaux ne sont pas identiques
    assert p1.weights[0] != p2.weights[0]

# ==========================================================================================
# 🔹 TEST 4 : Évaluation sur un dataset réel (sklearn)
# Vérifie que le modèle atteint une précision minimale sur un jeu de données généré.
# ==========================================================================================
def test_classification_accuracy():
    logger.info("Test: Dataset sklearn")
    
    from sklearn.datasets import make_classification
    
    # Génération d’un dataset binaire avec 2 features informatives
    X, y = make_classification(
        n_samples=100, n_features=2, n_classes=2,
        n_informative=2, n_redundant=0, random_state=42
    )
    
    # Transformation des étiquettes (0/1 → -1/1)
    y = np.where(y==0, -1, 1)
    
    # Entraînement du perceptron
    p = Perceptron(n_features=2, lr=0.1, epochs=20)
    p.fit(X, y)
    
    # Calcul de la précision
    acc = (p.predict(X) == y).mean()
    
    # Le modèle doit atteindre au moins 80% de précision
    assert acc > 0.8

# ==========================================================================================
# 🔹 TEST 5 : Gestion d’entrées invalides
# Vérifie que le modèle gère correctement les cas où les données sont vides.
# ==========================================================================================
def test_invalid_inputs():
    logger.info("Test: Données invalides")
    
    # Création du modèle
    p = Perceptron(n_features=2, lr=0.1)
    
    # On s’attend à une ValueError lors de l’entraînement sur des données vides
    with pytest.raises(ValueError):
        p.fit(np.array([]), np.array([]))

# ==========================================================================================
# 🔹 TEST 6 : Cas non linéairement séparable (XOR)
# Vérifie que le perceptron ne parvient pas à résoudre un problème non linéaire.
# ==========================================================================================
def test_non_separable_data():
    logger.info("Test: Cas XOR non séparable")
    
    # Jeu de données pour la fonction XOR (non linéairement séparable)
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([-1,1,1,-1])
    
    # Entraînement
    p = Perceptron(n_features=2, lr=0.1, epochs=50)
    p.fit(X, y)
    
    # Taux de réussite attendu < 100% (car le perceptron ne peut pas résoudre XOR)
    acc = (p.predict(X) == y).mean()
    assert acc < 1.0
