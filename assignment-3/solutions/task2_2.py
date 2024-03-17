# %%writefile task2_2.py
import csv
import json
import sys
import time

import numpy as np
from pyspark import SparkConf, SparkContext
from xgboost import XGBRegressor


def save_data(data, output_file_name):
    header = ["user_id", "business_id", "prediction"]
    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def read_csv_spark(path, sc):
    rdd = sc.textFile(path)
    header = rdd.first()
    rdd = rdd.filter(lambda row: row != header).map(lambda row: row.split(","))
    return rdd


def read_json_spark(path, sc):
    return sc.textFile(path).map(lambda row: json.loads(row))


def process_reviews(review_rdd):
    review_rdd = (
        review_rdd.map(
            lambda row: (row["business_id"], (float(row["useful"]), float(row["funny"]), float(row["cool"])))
        )
        .groupByKey()
        .mapValues(lambda x: tuple(sum(col) / len(col) for col in zip(*x)))
        .cache()
    )
    return review_rdd.collectAsMap()


def process_user(usr_rdd):
    usr_rdd = usr_rdd.map(
        lambda row: (row["user_id"], (float(row["average_stars"]), float(row["review_count"]), float(row["fans"])))
    ).cache()
    return usr_rdd.collectAsMap()


def process_bus(bus_rdd):
    bus_rdd = bus_rdd.map(lambda row: (row["business_id"], (float(row["stars"]), float(row["review_count"])))).cache()
    return bus_rdd.collectAsMap()


def process_train_data(row, review_dict, usr_dict, bus_dict):
    if len(row) == 3:
        usr, bus, rating = row
    else:
        usr, bus = row
        rating = None

    useful, funny, cool = review_dict.get(bus, (None, None, None))
    usr_avg_star, usr_review_cnt, usr_fans = usr_dict.get(usr, (None, None, None))
    bus_avg_star, bus_review_cnt = bus_dict.get(bus, (None, None))

    return ([useful, funny, cool, usr_avg_star, usr_review_cnt, usr_fans, bus_avg_star, bus_review_cnt], rating)


def task2_2(folder_path, test_file_name, output_file_name):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 2.2: : Model-based recommendation system")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        # Read and process the train data
        train_rdd = read_csv_spark(folder_path + "/yelp_train.csv", spark)

        review_rdd = read_json_spark(folder_path + "/review_train.json", spark)
        review_rdd = process_reviews(review_rdd)

        usr_rdd = read_json_spark(folder_path + "/user.json", spark)
        usr_rdd = process_user(usr_rdd)

        bus_rdd = read_json_spark(folder_path + "/business.json", spark)
        bus_rdd = process_bus(bus_rdd)

        # Read and process validation dataset
        val_rdd = read_csv_spark(test_file_name, spark).cache()

        # Train X and Y
        train_rdd = train_rdd.map(lambda x: process_train_data(x, review_rdd, usr_rdd, bus_rdd))

        # Valid x and Y
        val_processed = val_rdd.map(lambda x: process_train_data(x, review_rdd, usr_rdd, bus_rdd))

        # Extract X_train and Y_train
        X_train = train_rdd.map(lambda x: x[0]).cache()
        X_train = np.array(X_train.collect(), dtype="float32")
        Y_train = train_rdd.map(lambda x: x[1]).cache()
        Y_train = np.array(Y_train.collect(), dtype="float32")

        # Extract X_train and Y_train
        X_val = val_processed.map(lambda x: x[0]).cache()
        X_val = np.array(X_val.collect(), dtype="float32")
        # Y_val = val_processed.map(lambda x: x[1]).cache()
        # Y_val = np.array(Y_val.collect(), dtype='float32')

        xgb = XGBRegressor()
        xgb.fit(X_train, Y_train)
        Y_pred = xgb.predict(X_val)

        pred_data = []
        for i, row in enumerate(val_rdd.collect()):
            pred_data.append([row[0], row[1], Y_pred[i]])

        save_data(pred_data, output_file_name)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_1.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)

    # Read input parameters
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    task2_2(folder_path, test_file_name, output_file_name)

# task2_2("HW3StudentData", "HW3StudentData/yelp_val.csv", "t2_2.csv")
