import pyspark
import os
from pyspark.sql.functions import *
from pyspark.sql.types import *


'''
#################### DATA MANAGEMENT PIPELINE ####################

This pipeline prepares the data and outputs the matrix used in the
data anlysis pipeline

1. Extract data from DW
2. Extract sensor values from CSVs
3. Join the data from the two sources and take from the DW only the
aircrafts and days for which there's sensor value
4. Extract data from AMOS and filter on sensor 3453 to label each
tuple
5. Join and label each tuple: Maintenance or NonMaintenance

################################################################
'''


def process(spark, sc):

    print("Extract Data from DW")
    # Extract data from the DW (structured)
    aircraft_utilization = (spark.read.format("jdbc").option("driver","org.postgresql.Driver").option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require").option("dbtable", "public.aircraftutilization").option("user", "mireia.louzan").option("password", "DB161102").load())
    df = aircraft_utilization.select("aircraftid","timeid","flighthours","flightcycles","delayedminutes") # take only the needed data from the table
    df_dw = df.withColumn("timeid", df["timeid"].cast("String")) # in order to have the same type of data form timeid in DW - csv



    print("Extract Data from CSVs")
    # Extract data from the CSV (sensor info, semi-structured)
    path = './resources/trainingData'
    # Creation of an empty dataframe which will contain the sensor's info of all the flights
    emp_RDD = spark.sparkContext.emptyRDD()
    structure = StructType([StructField('TimeID2', StringType(), False), StructField('Value', IntegerType(), False), StructField('AircraftID2', StringType(), False)])
    first_df = spark.createDataFrame(data=emp_RDD,schema=structure)

    for file in os.scandir(path):
        # file contains measurements every 5 min from take-off to landing of a determinated flight
        aircraft_id = file.name[-10:-4] # aircratf id is contained in the csv file name
        flight_df = spark.read.csv(path + '/' + file.name, header = True, sep=';', inferSchema=True)
        df = flight_df.withColumn('TimeID2', concat(flight_df.date.substr(1, 10))).select("TimeID2", "Value").withColumn("AircraftID2", lit(aircraft_id)) # add the aircraft_id and the corresponding day to the dataframe
        first_df = first_df.union(df) # Put together the information of this flight with the information of flights collected until now

    # Aggregate the information by aircraft and day (compute the average)
    df_csv = first_df.groupby("AircraftID2", "TimeID2").avg("Value")


    # Join: data from the DW and data from the csv (aircraftid, timeid), and take only tuples with sensor data (inner join)
    pipeline_management = df_dw.join(df_csv, (df_dw.aircraftid == df_csv.AircraftID2) & (df_dw.timeid == df_csv.TimeID2)).select("aircraftid", "timeid", "flighthours", "flightcycles", "delayedminutes", "avg(Value)")


    print("Extract data from AMOS")
    # Extract data from AMOS (to label Maintenance/NonMaintenance)
    operation_interruption = (spark.read.format("jdbc").option("driver","org.postgresql.Driver").option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require").option("dbtable", "oldinstance.operationinterruption").option("user", "mireia.louzan").option("password", "DB161102").load())
    df_amos = operation_interruption.filter(operation_interruption.subsystem == "3453").select("aircraftregistration", "starttime") # Filter the aircrafts with information of the concret sensor and take only the needed columns
    df_amos = df_amos.withColumn("starttime", df_amos["starttime"].cast("String"))
    df_amos = df_amos.withColumn("starttime", concat(df_amos.starttime.substr(1, 10))) # set date in the correct format
    df_amos = df_amos.distinct() # distinct to delete duplicates and reduce the number of rows

    # Each maintenance has to be considerer from 6 days before, since we want to predict
    # if there will be an unexpected maintenance in the following 7 days for a given
    # day. So, for each maintenance, the previous 6 days has to be imputed as Maintenance
    # Add columns corresponding to the seven previous day an interruption took place
    df_amos = df_amos.withColumn("starttime-1", date_add(df_amos.starttime, -1))
    df_amos = df_amos.withColumn("starttime-2", date_add(df_amos.starttime, -2))
    df_amos = df_amos.withColumn("starttime-3", date_add(df_amos.starttime, -3))
    df_amos = df_amos.withColumn("starttime-4", date_add(df_amos.starttime, -4))
    df_amos = df_amos.withColumn("starttime-5", date_add(df_amos.starttime, -5))
    df_amos = df_amos.withColumn("starttime-6", date_add(df_amos.starttime, -6))


    # Make unions of the dataset with each of the columns corresponding to an added day
    # to get a tuple for each of the days that would correspond a maintenance.
    # Distinct after each of the unions to delete duplicate rows and reduce the dataframe size
    df_amos_f = df_amos.select("aircraftregistration", "starttime").union(df_amos.select("aircraftregistration", "starttime-1"))
    df_amos_f = df_amos_f.distinct()
    df_amos_f = df_amos_f.union(df_amos.select("aircraftregistration", "starttime-2"))
    df_amos_f = df_amos_f.distinct()
    df_amos_f = df_amos_f.union(df_amos.select("aircraftregistration", "starttime-3"))
    df_amos_f = df_amos_f.distinct()
    df_amos_f = df_amos_f.union(df_amos.select("aircraftregistration", "starttime-4"))
    df_amos_f = df_amos_f.distinct()
    df_amos_f = df_amos_f.union(df_amos.select("aircraftregistration", "starttime-5"))
    df_amos_f = df_amos_f.distinct()
    df_amos_f = df_amos_f.union(df_amos.select("aircraftregistration", "starttime-6"))
    df_amos_f = df_amos_f.distinct()

    # add a Label corresponding to the tuples with Maintenance
    df_amos_f = df_amos_f.withColumn("Label", lit('Maintenance'))

    # Join the result of the previous join with the AMOS data, to label the tuples
    # We want a Label for each of the tuples in 'sol', so we should do a "left" join
    # For the tuples which don't get a label after the join, imput the label NonMaintenance
    sol = pipeline_management.join(df_amos_f, (pipeline_management.aircraftid == df_amos_f.aircraftregistration) & (pipeline_management.timeid == df_amos_f.starttime), "left")
    solution = sol.withColumn("Label", when(sol.Label.isNull(), "NonMaintenance").otherwise(sol.Label)).select("flighthours", "flightcycles", "delayedminutes", "avg(Value)", "Label")

    # Save the generated matrix in a file named 'management_matrix'
    solution.write.save(path='management_matrix', format='csv', header=True, mode = 'overwrite')
