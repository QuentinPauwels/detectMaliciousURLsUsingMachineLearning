from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.classification import LogisticRegression


from pyspark.sql import SparkSession, functions

# Créer un objet SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# Charger les données à partir d'un fichier CSV
data = spark.read.csv("url_data.csv", header=True, inferSchema=True)

# convert the Class column to a numeric type
data = data.withColumn("Class", functions.when(data["Class"] == "good", 0)
                                         .otherwise(1))

# Décomposer les URL en mots individuels
tokenizer = Tokenizer(inputCol="URLs", outputCol="words")
data = tokenizer.transform(data)

# Compter le nombre d'occurrences de chaque mot
counter = CountVectorizer(inputCol="words", outputCol="features")
data = counter.fit(data).transform(data)

# Diviser les données en deux ensembles aléatoires
train_data, test_data = data.randomSplit([0.8, 0.2])

# Sélectionner les colonnes d'entrée et de sortie
train_input = train_data.select("features", "Class")
train_output = train_data.select("Class")

# Entraîner un modèle de forêts aléatoires
#rf = RandomForestClassifier(labelCol="Class", featuresCol="features", maxDepth=3, numTrees=2)
lr = LogisticRegression(labelCol="Class", featuresCol="features")

model = lr.fit(train_input)

# Utiliser le modèle pour faire des prédictions sur les données de test
test_input = test_data.select("features", "Class")
predictions = model.transform(test_input)

# Calculer l'erreur de classification
evaluator = MulticlassClassificationEvaluator(labelCol="Class", predictionCol="prediction")

# Afficher les résultats de l'évaluation
print("Accuracy = %g" % evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"}))
print("F1-Score = %g" % evaluator.evaluate(predictions, {evaluator.metricName: "f1"}))

# Afficher les prédictions en utilisant un diagramme à barres
predictions.select("Class", "prediction").show()


'''
# Sélectionner les colonnes d'entrée des données d'entraînement
input_cols = ["URLs"]
train_input = train_data.select(input_cols)

# Sélectionner la colonne de sortie des données d'entraînement
output_col = "target"
train_output = train_data.select(output_col)

# Sélectionner les colonnes d'entrée des données de test
test_input = test_data.select(input_cols)

# Sélectionner la colonne de sortie des données de test
test_output = test_data.select(output_col)

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer

# Assembler les colonnes d'entrée en un vecteur de caractéristiques
assembler = VectorAssembler(inputCols=input_cols, outputCol="features")
train_input = assembler.transform(train_input)
test_input = assembler.transform(test_input)

# Indexer la colonne de sortie en valeurs numériques
indexer = StringIndexer(inputCol=output_col, outputCol="label")
train_output = indexer.fit(train_output).transform(train_output)
test_output = indexer.fit(test_output).transform(test_output)

# Créer un objet RandomForestClassifier
classifier = RandomForestClassifier(labelCol="label", featuresCol="features")

# Entraîner le modèle sur les données d'entraînement
model = classifier.fit(train_input)

# Faire des prédictions sur les données de test
predictions = model.transform(test_input)

# Calculer l'erreur de classification
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g" % (accuracy))

# Afficher la matrice de confusion
predictions.groupBy("label", "prediction").count().show()'''
