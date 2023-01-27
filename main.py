import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession
import management
import analysis
import classifier
import argparse

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3"
PYSPARK_DRIVER_PYTHON = "python3"


'''
The main execution needs three flags
1. -pipeline must get a value in (management, analysis, classifier). If no value
is specified, the default is management, so it is not compulsory.
2. -aircraft is the value of the aircraft to predict. It is required.
3. -date is the value of the day to predict. It is also required.
'''


if(__name__== "__main__"):
    parser = argparse.ArgumentParser()
    # define and get from the execution line the three flags
    parser.add_argument('-pipeline', default='management')
    parser.add_argument('-aircraft', default=None, required=True)
    parser.add_argument('-date', default=None, required=True)
    args = parser.parse_args()

    pipeline = args.pipeline
    aircraft = args.aircraft
    date = args.date


    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder.config(conf=conf).master("local").appName("Training").getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()


    # Point each of the pipelines depending on the flag specified.
    print('Pipeline:', pipeline)
    if pipeline == 'management':
        # If the flag value is 'management', the three pipelines must be executed sequentially
        # Create the matrix containing the data from both sources (DW and CSVs)
        print('-----------------------------------------------------------------')
        print('DATA MANAGEMENT PIPELINE')
        management.process(spark, sc)
        # Train the model
        print('-----------------------------------------------------------------')
        print('DATA ANALYSIS PIPELINE')
        analysis.process(spark, sc)
        # Predict thr given record
        print('-----------------------------------------------------------------')
        print('RUN-TIME CLASSIFIER PIPELINE')
        classifier.process(spark, sc, aircraft, date)
    elif pipeline == 'analysis':
        # If the flag value is 'analysis', the management pipeline doesn't need to be executed
        # Train the model
        print('-----------------------------------------------------------------')
        print('DATA ANALYSIS PIPELINE')
        analysis.process(spark, sc)
        # Predict thr given record
        print('-----------------------------------------------------------------')
        print('RUN-TIME CLASSIFIER PIPELINE')
        classifier.process(spark, sc, aircraft, date)
    elif pipeline == 'classifier':
        # If the flag value is 'classifier', the given recors need to be predicted and the only pipeline is the third one
        # Predict thr given record
        print('-----------------------------------------------------------------')
        print('RUN-TIME CLASSIFIER PIPELINE')
        classifier.process(spark, sc, aircraft, date)
    else:
        # if the flag value is none of the specified, there's been a mistake
        print('Wrong pipeline')
