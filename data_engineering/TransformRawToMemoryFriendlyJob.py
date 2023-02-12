import numpy as np
import pandas as pd
import os
import shutil

from pathlib import Path
import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql.types import ArrayType, LongType, IntegerType



def convert_type_num(arr):
    """
    convert type string into 0(clicks), 1(carts), 2(orders).
    """
    return [0 if i == "clicks" else (1 if i == "carts" else 2) for i in arr]

def convert_ts_second_num(arr):
    """
    convert nanosecond timestamps to second.
    """
    return [int(i/1000) for i in arr]


def data_preprocessing(df_spark):
    """
    Preprocess the data, convert ts, action types and extract brief session features
    """
    convert_type_num_udf = udf(lambda row: convert_type_num(row), ArrayType(IntegerType()))
    convert_ts_seconds_udf = udf(lambda row:  convert_ts_second_num(row), ArrayType(LongType()))

    result_df = df_spark.withColumn("total_action", size(col("events")))\
                        .withColumn("aids", col("events.aid"))\
                        .withColumn("ts_seconds", convert_ts_seconds_udf(col("events.ts")) )\
                        .withColumn("action_types", convert_type_num_udf(col("events.type")))\
                        .withColumn("session_start_time", col("ts_seconds").getItem(0))\
                        .withColumn("session_end_time",element_at(col('ts_seconds'), -1))\
                        .select("session", "total_action", "session_start_time", "session_end_time", "aids", "ts_seconds", "action_types")

    return result_df


def save_csv_meta_info(df_spark, filename, path):
    """
    Method for saving the meta data info as csv
    """
    df_spark.select("session", "total_action", "session_start_time", "session_end_time").write.parquet(f"{path}/temp_file.parquet")
    df_core_pd = pd.read_parquet(f'{path}/temp_file.parquet', engine='pyarrow')
    df_core_pd.to_csv(f'{path}/{filename}', index = False)
    ## delete the temp file from disc
    if os.path.exists(f'{path}/temp_file.parquet'):
        shutil.rmtree(f'{path}/temp_file.parquet')
    else:
        print("Warnings: temp file removal error.")
        

def save_npz_core_info(df_spark, filename, path):
    """
    Method for saving the core action data, aid, ts, action_type as column-based data, in .npz format
    """
    ## save the core info, aids, ts, ops as .npz file
    ## Step I: convert to exploded pyspark dfs
    df_parquet = df_spark.select(explode(arrays_zip("aids", "ts_seconds", "action_types"))) \
                            .select("col.aids", "col.ts_seconds", "col.action_types")
    ## Step II: save temp parquet file to disc
    df_parquet.write.parquet(f"{path}/temp_file.parquet")
    ## step III: read the temp parquet file from disc
    df_core_pd = pd.read_parquet(f'{path}/temp_file.parquet', engine='pyarrow')
    np_aids = np.array(df_core_pd["aids"])
    np_ts = np.array(df_core_pd["ts_seconds"])
    np_ops = np.array(df_core_pd["action_types"])
    ## step IV: save the np arrays as .npz file
    np.savez(f"{path}/{filename}", aids=np_aids, ts=np_ts, ops=np_ops)
    ## delete the temp file from disc
    if os.path.exists(f'{path}/temp_file.parquet'):
        shutil.rmtree(f'{path}/temp_file.parquet')
    else:
        print("Warnings: temp file removal error.")


def main(train_input_path: Path,  test_input_path: Path, output_path: Path):
    spark = SparkSession.builder.appName("ottoDB").getOrCreate()

    trainDF = spark.read.json(str(train_input_path), lineSep='\n')
    testDF = spark.read.json(str(test_input_path), lineSep='\n')

    trainDF_transformed = data_preprocessing(trainDF)
    testDF_transformed = data_preprocessing(testDF)

    del trainDF, testDF

    ## save df data 
    save_csv_meta_info(trainDF_transformed, "train_meta_data.csv", str(output_path))
    save_csv_meta_info(testDF_transformed, "test_meta_data.csv", str(output_path))
    
    ## save column based data
    save_npz_core_info(trainDF_transformed, "train_core_data.npz", str(output_path))
    save_npz_core_info(testDF_transformed, "test_core_data.npz", str(output_path))



if __name__ == '__main__':
    startTime = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', type=Path, required=True)
    parser.add_argument('--test-data-path', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    args = parser.parse_args()
    main(args.train_data_path, args.test_data_path, args.output_path)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))   