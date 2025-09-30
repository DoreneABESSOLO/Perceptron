import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from perceptron import PerceptronModel
from sklearn.metrics import classification_report, confusion_matrix
    

def data_pre_processing(data):
    if 'Unnamed: 32' in data.columns:
        data.drop(columns=['Unnamed: 32'], inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    datacorr = data.corr()
    columns_to_keep = datacorr.index[abs(datacorr['diagnosis']) >= 0.4]
    filtered_data = data[columns_to_keep]
    
    filtered_data.reset_index(drop=True, inplace=True)
    
    X = filtered_data.drop(columns=['diagnosis'])
    y = filtered_data['diagnosis'].reset_index(drop=True)  
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, y

# la fonction prépare les données avant de les donner au modèle
#et garde les variables pertinentes (avec une corrélation supérieure à 0.4)
#puis réinitialise les index après le filtrage pour éviter les trous dans l'index
#sépare les features et la cible puis réinitialise aussi pour la correspondance à aux features
#normalisaion des features pour l'acp et réduction de dimmension en gardant assez de composante pour expliquer la variance



def main():
    data = pd.read_csv('bcw_data.csv')
    X_pca, y = data_pre_processing(data)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    y_train = y_train.values
    
    perceptron = PerceptronModel(learning_rate=0.1, max_iter=30)
    perceptron.fit(X_train, y_train)
    
    y_pred = perceptron.predict(X_test)
    accuracy, report, cm = perceptron.evaluate(y_test.values, y_pred) 
    # on appelle la fonction précédente pour obtenir les features réduites par l'acp et les étiquettes de y
    #ensuite on sépare les données pour l'entrainement et pour le test
    #On transforme le tableau en matrice 
    #on fait appel à la classe perceptron(pour entrainement) qui a été créé avec un pas de 0.1 et un max iter de 30
    #prédiction sur les données de test et éval des performances
    print("Accuracy:", accuracy)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)



if __name__=='__main__':
    main()