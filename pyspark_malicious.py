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
data = spark.read.csv("urldata.csv", header=True, inferSchema=True)

# convert the Class column to a numeric type
data = data.withColumn("Class", functions.when(data["Class"] == "benign", 0)
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

# Afficher la matrice de confusion
predictions.groupBy("Class", "prediction").count().show()
