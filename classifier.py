import pyspark
from pyspark.sql.types import *
from pyspark.sql.functions import *
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassificationModel


'''
#################### RUN-TIME CLASSIFIER PIPELINE ####################

This pipeline predicts if an aircraft is going to go for unscheduled
maintenance, given an aircraft and a day, and supposing there's data for
this concrete record.

To execute this pipeline, it is suposed to exist a file named
'analysis_tree', which contains the train model from the previous
pipeline execution.

1. Create the new record (<FH, FC, DM, AVG(sensor)>) by replicating the
management pipeline:
    - Extract the KPIs from the DW (FH, FC, DM) for the given aircraft
    and day
    - Extract the values of the sensor from the flights of the aircraft
    this day.
2. Prepare the tuple to be imputted into the trained model
3. Load the trained model (analysis_tree)
4. Classify the record and output the prediction

################################################################
'''


def process(spark, sc, aircraftID, day):
    # day: DD-MM-YYYY

    print("Extract Data from DW")
    # Extract data from the DW (structured), kpis that we want to take into account
    aircraft_utilization = (spark.read.format("jdbc").option("driver","org.postgresql.Driver").option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require").option("dbtable", "public.aircraftutilization").option("user", "@user_name").option("password", "PASSWORD_CONNECTION").load())
    df = aircraft_utilization.select("aircraftid","timeid","flighthours","flightcycles","delayedminutes")
    df = df.withColumn("timeid", df["timeid"].cast("String"))
    df_info = df.filter((df.aircraftid == aircraftID) & (df.timeid == day)) # filter the data for the given aircraft and date

    # the date in the csv file name is in YYMMDD format, so we need to create this from the given day
    day2 = day[8]+day[9]+day[5]+day[6]+day[2]+day[3]

    # add avgerage sensor info
    print("Extract Data from CSVs")
    # Extract data from the CSV (sensor info, semi-structured)
    path = './resources/trainingData'

    # Creation of an empty dataframe which will contain the sensor's info of the aircarftID in day
    emp_RDD = spark.sparkContext.emptyRDD()
    structure = StructType([StructField('Value', DoubleType(), False)])
    first_df = spark.createDataFrame(data=emp_RDD,schema=structure)

    # iterate df looking for a match with aircraftID and day with aircraftid and timeid
    for file in os.scandir(path):
        # file contains measurements every 5 min from take-off to landing of a determinated flight
        aircraft_id = file.name[-10:-4] # aircratf id is contained in the csv file name
        time = file.name[0:6] # day is contained in the csv file name
        if aircraft_id == aircraftID and time == day2:
            flight_df = spark.read.csv(path + '/' + file.name, header = True, sep=';', inferSchema=True)
            # for this pipeline we only need the value column, since the aircraft and date will be the same for all of them
            flight_df = flight_df.select("Value")
            first_df = first_df.union(flight_df)

    # Average the sensor value
    first_df = first_df.select(mean('Value'))
    # Take the avg value as a variable for being able to input it
    avg = first_df.select('avg(Value)').first()['avg(Value)']

    # Tuple pepared to be inputatted
    print('TuplePrepared')
    df_info = df_info.withColumn('avg(Value)', lit(avg)).select("flighthours", "flightcycles", "delayedminutes", "avg(Value)")

    # Load the trained model
    model = DecisionTreeClassificationModel.load("analysis_tree")

    # change data format to the needed by the model to predict
    features = ["flighthours", "flightcycles", "delayedminutes", "avg(Value)"]
    va = VectorAssembler(inputCols=features, outputCol="features").transform(df_info)

    # compute predictions
    predictions = model.transform(va)

    prediction = predictions.select('prediction').first()['prediction']

    # Output Maintenance or NonMaintenance based on the prediction
    if prediction == 0:
        print("Predicted: NonMaintenance")
    else:
        print("Predicted: Maintenance")
