# Perceptron
## 1) Initialisation Spark
``` bash
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col

spark = SparkSession.builder.appName("BCWPerceptron").getOrCreate()
```


``` bash 
from pyspark.sql import SparkSession
```
 importe la classe pour créer une session Spark (point d’entrée vers l’API Spark).

``` bash 
from pyspark.sql.functions import when, col
 ```
 importe des fonctions utilitaires Spark SQL (when pour conditions, col pour référencer une colonne).

``` bash 
spark = SparkSession.builder...getOrCreate()
``` 
 crée (ou récupère) une SparkSession nommée "BCWPerceptron". C’est nécessaire pour lire des fichiers et exécuter des transformations.

## 2) Chargement et nettoyage des données
``` bash
df = spark.read.csv("bcw_data.csv", header=True, inferSchema=True)
```
Charge le CSV bcw_data.csv.
``` bash 
header=True
``` 
 indique qu’il y a une ligne d’en-tête ;
``` bash
inferSchema=True
```
 demande à Spark d’essayer de deviner les types de colonnes.
``` bash
df = df.drop("id", "Unnamed: 32")
```
 Supprime les colonnes inutiles id et la colonne vide Unnamed: 32 (la colonne « vide » qui apparaît souvent avec ce dataset).

``` bash
df = df.withColumn("label", when(col("diagnosis") == "M", 1).otherwise(-1))
```
 Crée une nouvelle colonne label : si diagnosis == "M" → 1 (malin), sinon -1 (bénin). On encode ainsi la cible pour notre perceptron (signe +1 / -1).
```bash
df = df.drop("diagnosis")
```
 Supprime la colonne diagnosis car elle est maintenant remplacée par label.
``` bash
bad_cols = [c for c, t in df.dtypes if t not in ("double", "int")]
print("Colonnes non numériques détectées:", bad_cols)
df = df.drop(*bad_cols)
```
df.dtypes renvoie la liste des paires (nom_colonne, type). On filtre pour trouver les colonnes dont le type n’est ni double ni int (ex. colonnes string restant).

On affiche les colonnes non numériques détectées (debug).

On les supprime pour garantir que seules des colonnes numériques restent (important pour VectorAssembler).

## 3) Assemblage des features
``` bash
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType
```

 Import des utilitaires : VectorAssembler pour concaténer plusieurs features en un vecteur, udf pour définir des UDFs, et types Spark pour déclarer le retour d’UDF.
``` bash
feature_cols = [c for c in df.columns if c != "label"]
```

 Liste toutes les colonnes qui serviront de features (toutes moins label).

```bash
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
```
 Crée un assembleur qui prend les colonnes numériques feature_cols et produit une colonne features_vec de type Vector (Spark ML Vector, ex. DenseVector).

``` bash
def to_array(v):
    return [float(x) for x in v]

to_array_udf = udf(to_array, ArrayType(DoubleType()))
```
 Définition d’une fonction Python qui convertit un DenseVector Spark (ou séquence) en liste de float (format compatible numpy).

to_array_udf est un UDF déclaré comme retournant Array[Double].

``` bash
df = assembler.transform(df) \
              .withColumn("features", to_array_udf(col("features_vec"))) \
              .select("label", "features")
```

assembler.transform(df) ajoute features_vec (vecteur Spark).

.withColumn("features", to_array_udf(...)) convertit features_vec en liste python features.

.select("label", "features") ne garde que les colonnes nécessaires pour l’entraînement (label + features), simplifiant le DataFrame.

Remarque : on convertit en liste Python parce que notre code de perceptron utilise numpy dans les UDF/RDD map. On pourrait aussi manipuler les Vectors Spark directement, mais la conversion rend l’interaction avec numpy plus simple.

## 4) Implémentation du Perceptron (Estimator + Model)
``` bash 
imports et utilitaires
import numpy as np
from pyspark.sql.functions import lit
from pyspark.sql.types import IntegerType
from pyspark.ml import Estimator, Model, Pipeline
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import Param, Params, TypeConverters

```
numpy pour calcul vectoriel.

lit pour créer une colonne constante.

IntegerType pour typer les sorties d’UDFs de prédiction.

Estimator, Model et classes utilitaires pour intégrer le perceptron dans l’API Spark ML (Pipeline, sauvegarde/chargement, Param).
``` bash
PerceptronModel (Transformer)
class PerceptronModel(Model, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, weights, bias):
        super(PerceptronModel, self).__init__()
        self.weights = weights
        self.bias = bias

    def _predict_udf(self):
        w = self.weights
        b = self.bias
        def predict(features):
            return 1 if np.dot(w, features) + b >= 0 else -1
        return udf(predict, IntegerType())

    def _transform(self, df):
        predict_udf = self._predict_udf()
        return df.withColumn("prediction", predict_udf(col("features")))

```
class PerceptronModel(Model, ...) : c’est le Transformer qui applique le modèle entraîné.

__init__(self, weights, bias) : stocke weights (vecteur numpy) et bias (scalaire).

_predict_udf(self) : crée une UDF de prédiction qui, pour chaque features (liste de floats), calcule np.dot(w, features) + b et renvoie 1 si ≥ 0 sinon -1. Le udf(...) est typé IntegerType.

_transform(self, df) : méthode appelée par l’API Spark ML pour transformer un DataFrame donné ; elle ajoute la colonne prediction en appliquant l’UDF.

Remarque : DefaultParamsReadable / DefaultParamsWritable rendent le modèle sauvegardable via .write().save(...).
``` bash
PerceptronClassifier (Estimator)
class PerceptronClassifier(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    lr = Param(Params._dummy(), "lr", "learning rate", typeConverter=TypeConverters.toFloat)
    epochs = Param(Params._dummy(), "epochs", "number of epochs", typeConverter=TypeConverters.toInt)

    def __init__(self, lr=0.1, epochs=10):
        super(PerceptronClassifier, self).__init__()
        self._setDefault(lr=0.1, epochs=10)
        self._set(lr=lr, epochs=epochs)
```

PerceptronClassifier hérite d’Estimator : c’est la classe qui va entraîner et retourner un PerceptronModel.

lr et epochs sont des Param (paramètres configurables).

__init__ : initialise les paramètres, fixe les valeurs par défaut et applique les valeurs passées à la création.
``` bash
    def _fit(self, df):
        lr = self.getOrDefault(self.lr)
        epochs = self.getOrDefault(self.epochs)

        dim = len(df.first()["features"])
        w = np.zeros(dim)
        b = 0.0

```
_fit(self, df) : méthode qui réalise l’apprentissage. Elle doit retourner un Model.

lr et epochs : récupère les valeurs de paramètres.

dim = len(df.first()["features"]) : récupère la dimension du vecteur de features en lisant la première ligne (action). Attention : df.first() déclenche un job Spark.

w = np.zeros(dim) et b = 0.0 : initialisation des poids et du biais.

Pour chaque époque :
``` bash
        for epoch in range(epochs):
            def predict_local(features):
                return 1 if np.dot(w, features) + b >= 0 else -1
            predict_udf = udf(predict_local, IntegerType())
            df_pred = df.withColumn("prediction", predict_udf(col("features")))

```
À chaque itération on :

définit predict_local qui utilise w et b actuels (fermés dans la closure).

crée un predict_udf pour appliquer la règle sign sur chaque ligne du DataFrame.

df_pred ajoute la colonne prediction.
``` bash
            df_err = df_pred.withColumn(
                "error", when(col("label") * col("prediction") <= 0, lit(1)).otherwise(lit(0))
            )
```

df_err ajoute une colonne error valant 1 si l’exemple est mal classé (label * prediction <= 0) sinon 0. (Nous utilisons la condition <= 0 pour couvrir le cas d’égalité).
``` bash
            updates = df_err.rdd.map(lambda row: (
                row["error"] * row["label"],  
                np.array(row["features"]) if row["error"] == 1 else np.zeros(dim)
            ))

```
Convertit le DataFrame en RDD puis map pour produire pour chaque ligne un tuple :

premier élément : contribution au biais = error * label (valeur 0 si pas d’erreur, sinon label).

second élément : vecteur de mise à jour des poids = features si erreur, sinon vecteur nul.

Remarque : on crée un np.array à partir de la liste de features pour faciliter l’addition vectorielle ensuite.
``` bash
            total_update = updates.reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
```

reduce : agrège localement tous les tuples (bias_contrib, weight_contrib) en une somme globale (sum_bias_contrib, sum_weight_contrib).
``` bash
            b += lr * total_update[0]
            w += lr * total_update[1]

            print(f"Epoch {epoch+1}/{epochs} -> erreurs={total_update[0]}")

```
Met à jour le biais b et le vecteur w selon la règle perceptron : w = w + lr * sum(y*x) et b = b + lr * sum(y), où les sommes ne prennent en compte que les exemples mal classés.

Affiche un message de suivi avec le nombre d’erreurs agrégé (utile pour le debug et la convergence).
``` bash
        return PerceptronModel(weights=w, bias=b)
```

Après toutes les époques, retourne un PerceptronModel initialisé avec les poids appris.

5) Pipeline et entraînement
``` bash
perceptron = PerceptronClassifier(lr=0.01, epochs=20)
pipeline = Pipeline(stages=[perceptron])

model = pipeline.fit(df)

```
Instancie l’estimateur PerceptronClassifier avec lr=0.01 et 20 époques.

Crée un Pipeline avec une seule étape (le perceptron) — cela facilite l’intégration dans des workflows Spark ML.

model = pipeline.fit(df) : lance l’entraînement ; Spark appelle _fit() de PerceptronClassifier et récupère le PerceptronModel.
``` bash
df_pred = model.transform(df)
df_pred.show(10)
```

Applique le modèle entraîné au DataFrame (méthode _transform du PerceptronModel), ajoute la colonne prediction.

Affiche les 10 premières lignes de df_pred.

6) Évaluation : Accuracy
``` bash
from pyspark.sql.functions import avg

accuracy = df_pred.withColumn(
    "correct", when(col("label") == col("prediction"), 1).otherwise(0)
).agg(avg("correct")).collect()[0][0]

print(f" Accuracy = {accuracy*100:.2f}%")
```

withColumn("correct", ...) : ajoute 1 si prédiction correcte, 0 sinon.

.agg(avg("correct")) calcule la moyenne des correct → fraction des exemples correctement classés (accuracy).

.collect()[0][0] récupère la valeur du driver (action).

Affiche l’accuracy formatée.
