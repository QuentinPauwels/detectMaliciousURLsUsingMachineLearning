from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
import numpy as np
from pyspark.sql.types import ArrayType, StringType, IntegerType
import re
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier

# create a SparkConf instance
conf = SparkConf().setAppName("Malicious URL Detection")

# create a SparkContext instance
sc = SparkContext(conf=conf)

# set the SparkContext instance as the variable sc
sc = SparkContext.getOrCreate()

# create a SparkSession instance
spark = SparkSession(sc)

# load the URL data as a DataFrame
url_data = spark.read.csv("url_data.csv", header=True, inferSchema=True)

# Filter the DataFrame to only include rows where the "Class" column does not have a value of "good" or "bad"
filtered_df = url_data.filter(url_data["Class"].isin(["good", "bad"]))

# Print the filtered DataFrame
#filtered_df.show()

# Get the total number of rows in the DataFrame, including null values
num_rows = filtered_df.count()

# Print the number of rows
print(num_rows)

# convert the Class column to a numeric type
url_data = filtered_df.withColumn("ClassNumber", functions.when(filtered_df["Class"] == "good", 0)
                                         .otherwise(1))


url_data.show(1)

def my_tokenizer(url):  
  # Split by slash (/) and dash (-)
  tokens = re.split('[/-]', url)
  #print(tokens)
  for i in tokens:
    # Include the splits extensions and subdomains
    if i.find(".") >= 0:
      dot_split = i.split('.')
      
      # Remove .com and www. since they're too common
      if "com" in dot_split:
        dot_split.remove("com")
      if "www" in dot_split:
        dot_split.remove("www")
      
      tokens += dot_split
      
  return tokens


url_data = url_data.withColumn("ZBI", url_data.ClassNumber * 2)

udf = functions.UserDefinedFunction(my_tokenizer, returnType=ArrayType(StringType()))

# apply the UserDefinedFunction to the input column
output_column = udf(url_data["URLs"])
url_data = url_data.withColumn("tokens", output_column)

url_data = url_data.select("URLs", "Class", "ClassNumber", "tokens")

# Display the resulting DataFrame
url_data.show(1, truncate=False)

#firstTokens = url_data.select("tokens").take(1)
#print(firstTokens[0][0])

print(url_data.columns)
print(url_data.dtypes)

'''
# create an instance of the HashingTF class
hashingTF = HashingTF() \
                .setNumFeatures(150) \
                .setInputCol("tokens") \
                .setOutputCol("rawFeatures")

'''
# create a CountVectorizer instance
vectorizer = CountVectorizer(inputCol="tokens", outputCol="rawFeatures")

# create a LogisticRegression instance
lr = LogisticRegression(labelCol="ClassNumber", featuresCol="rawFeatures")

# create an instance of the DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="ClassNumber", featuresCol="rawFeatures", maxDepth=1)

rf = RandomForestClassifier(labelCol="ClassNumber", featuresCol="rawFeatures", numTrees=10)

# create a Pipeline instance with the stages: tokenizer, hashingTF, idf, lr
pipeline = Pipeline(stages=[vectorizer, lr])

# split the data into training and testing sets
training_data, testing_data = url_data.randomSplit([0.8, 0.2])

# fit the pipeline to the training data
model = pipeline.fit(training_data)

# make predictions on the testing data
predictions = model.transform(testing_data)

# print the predictions
predictions.show()

# filter the predictions to only show malicious URLs
malicious_urls = predictions.filter(predictions["prediction"] == 1)

# print the malicious URLs
malicious_urls.show()

# import the BinaryClassificationEvaluator class
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# create an instance of the BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator() \
                .setLabelCol("ClassNumber") \
                .setRawPredictionCol("prediction")

# compute the accuracy of the model
accuracy = evaluator.evaluate(predictions)

# print the accuracy of the model
print(accuracy)

# stop the SparkContext
sc.stop()