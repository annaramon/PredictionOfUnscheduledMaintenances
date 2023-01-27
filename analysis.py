from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import *
from sklearn.metrics import confusion_matrix


'''
#################### DATA ANALYSIS PIPELINE ####################

This pipeline trains a classifier (decision tree) based on the
data matrix created in the previous pipeline

To execute this pipeline, it is suposed to exist a file named
'management_matrix', which contains the matrix with the tuples to
train a model.

1. Load the matrix (management_matrix)
2. Perpare the data in the format to train the Model
3. Train the model
4. Compute some metrics to analyze the goodness of the model

################################################################
'''


def process(spark, sc):
    # Load the data stored in csv format as a DataFrame, matrix from the management pipeline
    data = spark.read.csv("management_matrix", header='True', inferSchema='True')

    # transform data into the accurate types because they are read as strings
    data = data.withColumn("flighthours",data["flighthours"].cast(DoubleType())) \
    .withColumn("flightcycles",data["flightcycles"].cast(IntegerType())) \
    .withColumn("delayedminutes",data["delayedminutes"].cast(DoubleType())) \
    .withColumn("avg(Value)",data["avg(Value)"].cast(DoubleType())) \
    .withColumn("Label",when(data["Label"] == "NonMaintenance",0).when(data["Label"] == "Maintenance", 1)) # transform the label into binari form


    # change data format to the needed by the model to train
    features = ["flighthours", "flightcycles", "delayedminutes", "avg(Value)"]
    va = VectorAssembler(inputCols=features, outputCol="features").transform(data)
    # aggregate all the features of which depend the Maintenance or not state
    va_df = va.select(["features", "Label"])


    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = va_df.randomSplit([0.7, 0.3])



    # Train a DecisionTree model
    dt = DecisionTreeClassifier(labelCol="Label", featuresCol="features")
    # Train model
    model = dt.fit(trainingData)

    # Make predictions
    predictions = model.transform(testData)


    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    # Print a confusion matrix in order to find out if the predictions are good enough
    y_pred = predictions.select("prediction").collect()
    y_orig = predictions.select("Label").collect()

    cm = confusion_matrix(y_orig, y_pred)
    print("Confusion Matrix:")
    print(cm)
    #tnm = true NonMaintenance; fm = false Maintenance; fnm = false NonMaintenance; tm = true Maintenance
    tnm, fm, fnm, tm = cm.ravel()
    # compute some metrics to see the goodness of fit
    # Prediction Accuracy
    print("Model's Prediction Accuracy:", accuracy)
    # Prediction error
    print("Test Error = %g" % (1.0 - accuracy))
    # Precision of both labels
    print('NonMaintenance precision:', tnm/(tnm+fnm))
    print('Maintenance precision:', tm/(tm+fm))
    # Recall for both labels
    print('NonMaintenance recall:', tnm/(tnm+fm))
    print('Maintenance recall:', tm/(tm+fnm))


    # Save the trained tree, the model
    model.write().overwrite().save("analysis_tree")
