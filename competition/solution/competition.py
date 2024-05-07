#######################################################################################################################
# Method Description
#######################################################################################################################
#
# **Model-Based Recommendation System for Yelp Reviews Data**

# This script implements a model-based recommendation system for Yelp reviews data. It aims to predict user ratings for
# businesses based on various features extracted from the Yelp Reviews dataset.

# ## Model
# To enhance the effectiveness of our recommendation system, I opted for a model-based approach over the hybrid method
# utilized in our previous work (HW3). This decision stems from the superior performance of model-based systems across
# various components of the recommendation process. I employ the XGBoost Regressor in this implementation, utilizing
# parameters optimized through local GridSearch.

# ## Feature Engineering:
# Our feature selection process draws from diverse aspects of the Yelp dataset, encompassing user-centric metrics
# (e.g., review count, average stars, elite status), business-oriented attributes (e.g., review count, average stars,
# attributes), and the dynamics of user-business interactions (e.g., reviews, tips). Additionally, I consider
# metadata from business-associated photos as an indicator of authenticity and trustworthiness. Throughout feature
# extraction, we employ aggregation techniques to consolidate sparse matrices and mitigate null entries.

# After extensive experimentation with different feature sets, we made strategic decisions regarding feature inclusion.
# While initially excluding review-related features due to their absence for (user, business) pairs in validation and
# test sets, reintroducing them became imperative after observing adverse effects on RMSE upon removal. Similarly,
# their reinstatement was necessary to maintain model performance despite the initial exclusion of tip and photo
# features.
#

#######################################################################################################################
# Error Distribution
#######################################################################################################################
#
# >=0 and <1:    102162
# >=1 and <2:     32993
# >=2 and <3:      6116
# >=3 and <4:       773
# >=4:                0
#

#######################################################################################################################
# RMSE
#######################################################################################################################
#
# Validation Set:         0.9772555489644983, good
# Test Set:               0.9760147801166215, good
#

#######################################################################################################################
# Execution Time
#######################################################################################################################
#
# Data Processing Time:       183.61311721801758 s
# Model Training Time:        425.02053117752075 s
# -----------------------------------------------------------
# Total Execution Time:       622.1228864192963 s
#


import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor


class Path:
    yelp_train_processed: str = "yelp_train_processed.csv"
    yelp_val_processed: str = "yelp_val_processed.csv"


class DataReader:
    def __init__(self, sc: SparkContext):
        self.sc = sc

    def read_csv_spark(self, path: str):
        rdd = self.sc.textFile(path)
        header = rdd.first()
        rdd = rdd.filter(lambda row: row != header).map(lambda row: row.split(","))
        return rdd, header.split()

    def read_json_spark(self, path: str):
        return self.sc.textFile(path).map(lambda row: json.loads(row))


class BusinessData:
    keys_to_delete = [
        "name",
        "neighborhood",
        "address",
        "attributes",
        "categories",
        "hours",
        "postal_code",
        "city",
        "state",
    ]

    @staticmethod
    def parse_row(row: dict):
        row["num_attrs"] = len(row["attributes"]) if row["attributes"] is not None else 0
        row["num_categories"] = len(row["categories"].split(",")) if row["categories"] is not None else 0
        # row["latitude"] = float(row["latitude"])
        # row["longitude"] = float(row["longitude"])
        row["stars"] = float(row["stars"])

        # Delete keys
        # row = {k: v for k, v in row.items() if k not in BusinessData.keys_to_delete}
        return row

    @staticmethod
    def generate_mapping(rdd):
        # Extract unique values for state and city
        state_to_index = rdd.map(lambda row: row["state"]).distinct().zipWithIndex().collectAsMap()
        city_to_index = rdd.map(lambda row: row["city"]).distinct().zipWithIndex().collectAsMap()

        return state_to_index, city_to_index

    def process(rdd):
        rdd = rdd.map(lambda row: BusinessData.parse_row(row)).map(
            lambda row: (
                row["business_id"],
                (
                    row["stars"],
                    row["review_count"],
                    row["is_open"],
                    row["num_attrs"],
                    row["num_categories"],
                    # row["latitude"],
                    # row["longitude"],
                ),
            )
        )
        return rdd.cache().collectAsMap()


class UserData:
    keys_to_delete = [
        "name",
        "friends",
        "elite",
        "yelping_since",
        "compliment_hot",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_cool",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
    ]
    compliment_keys = [
        "compliment_hot",
        "compliment_more",
        "compliment_profile",
        "compliment_cute",
        "compliment_list",
        "compliment_note",
        "compliment_plain",
        "compliment_cool",
        "compliment_funny",
        "compliment_writer",
        "compliment_photos",
    ]
    lc = len(compliment_keys)

    @staticmethod
    def parse_row(row: dict):
        row["num_elite"] = len(row["elite"].split(",")) if row["elite"] != "None" else 0
        row["num_friends"] = len(row["friends"].split(",")) if row["friends"] != "None" else 0
        row["avg_compliment"] = sum(row[key] for key in UserData.compliment_keys) / UserData.lc

        yelping_since = datetime.strptime(row["yelping_since"], "%Y-%m-%d")
        membership_years = datetime.now() - yelping_since
        row["membership_years"] = membership_years.days / 365.25

        row["average_stars"] = float(row["average_stars"])

        # Delete keys
        # row = {k: v for k, v in row.items() if k not in UserData.keys_to_delete}
        return row

    def process(rdd):
        rdd = rdd.map(lambda row: UserData.parse_row(row)).map(
            lambda row: (
                row["user_id"],
                (
                    row["review_count"],
                    row["useful"],
                    row["funny"],
                    row["cool"],
                    row["fans"],
                    row["average_stars"],
                    row["num_elite"],
                    row["num_friends"],
                    row["avg_compliment"],
                    row["membership_years"],
                ),
            )
        )
        return rdd.cache().collectAsMap()


class ReviewData:
    keys_to_delete = ["review_id", "date", "text"]

    @staticmethod
    def parse_row(row):

        # Delete keys
        row = {k: v for k, v in row.items() if k not in ReviewData.keys_to_delete}
        return row

    def process(rdd):
        rdd = (
            rdd
            # .map(lambda row: ReviewData.parse_row(row))
            .map(
                lambda row: (
                    (row["user_id"], row["business_id"]),
                    (row["stars"], row["useful"], row["funny"], row["cool"], 1),
                )
            )
            .reduceByKey(
                lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4])  # Sum values and counts
            )
            .mapValues(
                lambda values: (
                    values[0] / values[4],
                    values[1] / values[4],
                    values[2] / values[4],
                    values[3] / values[4],
                )  # Calculate averages
            )
        )
        return rdd.cache().collectAsMap()


class TipData:
    keys_to_delete = ["date", "text"]

    @staticmethod
    def parse_row(row):
        # Delete keys
        row = {k: v for k, v in row.items() if k not in TipData.keys_to_delete}
        return row

    def process(rdd):
        rdd = (
            rdd
            # .map(lambda row: TipData.parse_row(row))
            # Calculate sum of likes and review count by user
            .map(lambda row: ((row["user_id"], row["business_id"]), (row["likes"], 1))).reduceByKey(
                lambda x, y: (x[0] + y[0], x[1] + y[1])
            )
        )
        return rdd.cache().collectAsMap()


class PhotoData:
    keys_to_delete = ["photo_id", "caption"]
    possible_labels = ["drink", "food", "inside", "menu", "outside"]

    @staticmethod
    def parse_row(row):
        # Delete keys
        row = {k: v for k, v in row.items() if k not in PhotoData.keys_to_delete}
        return row

    def process(rdd):
        rdd = (
            rdd
            # .map(lambda row: PhotoData.parse_row(row))
            .map(lambda row: ((row["business_id"]), ([row["label"]], 1)))
            .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
            .mapValues(
                lambda values: (
                    # {label: values[0].count(label) for label in PhotoData.possible_labels},
                    len(set(values[0])),
                    values[1],
                )
            )
        )
        return rdd.cache().collectAsMap()


class ModelBasedConfig:
    drop_cols: list = [
        "user_id",
        "business_id",
        "rating",
        "review_avg_stars",
        "useful",
        "funny",
        "cool",
        # "num_attrs",
        # "num_categories"
        # "likes", "upvotes",
        # "num_cat", "num_img"
    ]
    params: dict = {
        "lambda": 9.92724463758443,
        "alpha": 0.2765119705933928,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "learning_rate": 0.02,
        "max_depth": 17,
        "random_state": 2020,
        "min_child_weight": 101,
        "n_estimators": 300,
    }
    pred_cols: list = ["user_id", "business_id", "prediction"]


def create_dataset(row, usr_dict, bus_dict, review_dict, tip_dict, img_dict):
    if len(row) == 3:
        usr, bus, rating = row
    else:
        usr, bus = row
        rating = None

    # From review_train.json
    r_avg_stars, useful, funny, cool = review_dict.get((usr, bus), (None, 0, 0, 0))

    # From user.json
    (
        usr_review_count,
        usr_useful,
        usr_funny,
        usr_cool,
        usr_fans,
        usr_avg_stars,
        num_elite,
        num_friends,
        usr_avg_comp,
        membership_years,
    ) = usr_dict.get(usr, (0, None, None, None, 0, 3.5, 0, 0, 0, None))

    # From business.json
    bus_avg_stars, bus_review_count, bus_is_open, num_attrs, num_categories = bus_dict.get(
        bus, (3.5, 0, None, None, None)
    )

    # From tip.json
    likes, upvotes = tip_dict.get((usr, bus), (0, 0))

    # From photo.json
    num_cat, num_img = img_dict.get(bus, (0, 0))

    return (
        usr,
        bus,
        r_avg_stars,
        useful,
        funny,
        cool,
        usr_review_count,
        usr_useful,
        usr_funny,
        usr_cool,
        usr_fans,
        usr_avg_stars,
        num_elite,
        num_friends,
        usr_avg_comp,
        membership_years,
        bus_avg_stars,
        bus_review_count,
        bus_is_open,
        num_attrs,
        num_categories,
        likes,
        upvotes,
        num_cat,
        num_img,
        float(rating),
    )


def save_data(data: list, output_file_name: str):
    header = ["user_id", "business_id", "prediction"]
    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def process_data(folder_path: str, test_file_name: str):
    start_time = time.time()

    # Initialize Spark
    conf = SparkConf().setAppName("Competition: Recommendation system")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        data_reader = DataReader(spark)

        # Additional Data
        # User related data
        usr_rdd = data_reader.read_json_spark(os.path.join(folder_path, "user.json"))
        usr_rdd = UserData.process(usr_rdd)

        # Business related data
        bus_rdd = data_reader.read_json_spark(os.path.join(folder_path, "business.json"))
        bus_rdd = BusinessData.process(bus_rdd)

        # User to Business Reviews
        review_rdd = data_reader.read_json_spark(os.path.join(folder_path, "review_train.json"))
        review_rdd = ReviewData.process(review_rdd)

        # User 2 Business Tip
        tip_rdd = data_reader.read_json_spark(os.path.join(folder_path, "tip.json"))
        tip_rdd = TipData.process(tip_rdd)

        # Business Photo Data
        img_rdd = data_reader.read_json_spark(os.path.join(folder_path, "photo.json"))
        img_rdd = PhotoData.process(img_rdd)

        # Business checkin data
        # cin_rdd = data_reader.read_json_spark(os.path.join(folder_path, "checkin.json"))

        # Read train dataset
        train_rdd, _ = data_reader.read_csv_spark(os.path.join(folder_path, "yelp_train.csv"))

        # Read validation dataset
        # test_file_name = os.path.basename(test_file_name)
        val_rdd, _ = data_reader.read_csv_spark(test_file_name)

        # Merge datasets
        train_processed = train_rdd.map(lambda row: create_dataset(row, usr_rdd, bus_rdd, review_rdd, tip_rdd, img_rdd))
        val_processed = val_rdd.map(lambda row: create_dataset(row, usr_rdd, bus_rdd, review_rdd, tip_rdd, img_rdd))

        # Convert processed datasets to Pandas DataFrame and save as CSV file
        column_names = [
            "user_id",
            "business_id",
            "review_avg_stars",
            "useful",
            "funny",
            "cool",
            "usr_review_count",
            "usr_useful",
            "usr_funny",
            "usr_cool",
            "usr_fans",
            "usr_avg_stars",
            "num_elite",
            "num_friends",
            "usr_avg_comp",
            "membership_years",
            "bus_avg_stars",
            "bus_review_count",
            "bus_is_open",
            "num_attrs",
            "num_categories",
            "likes",
            "upvotes",
            "num_cat",
            "num_img",
            "rating",
        ]

        train_df_processed = pd.DataFrame(train_processed.collect(), columns=column_names)
        train_df_processed.to_csv(Path.yelp_train_processed, index=False)

        val_df_processed = pd.DataFrame(val_processed.collect(), columns=column_names)
        val_df_processed.to_csv(Path.yelp_val_processed, index=False)

    except Exception as e:
        print(f"Exception occured:\n{e}")

    finally:
        spark.stop()

    execution_time = time.time() - start_time
    print(f"Data Processing Duration: {execution_time} s\n")


def train_model(train_data_path: str, test_data_path: str):
    start_time = time.time()
    # Read processed data
    train_df_processed = pd.read_csv(train_data_path)
    val_df_processed = pd.read_csv(test_data_path)

    # Initialize min-max scaler
    scaler = MinMaxScaler()

    # Apply min-max normalization to the training and test data
    X_train_norm = scaler.fit_transform(train_df_processed.drop(columns=ModelBasedConfig.drop_cols))
    y_train = train_df_processed["rating"]
    X_test_norm = scaler.transform(val_df_processed.drop(columns=ModelBasedConfig.drop_cols))
    # y_test = val_df_processed["rating"]

    # Train XGBoost Regression Model
    model = XGBRegressor(**ModelBasedConfig.params)
    model.fit(X_train_norm, y_train)

    # Predict Rating on Test Data
    y_test_pred = model.predict(X_test_norm)
    val_df_processed["prediction"] = y_test_pred

    # Filter Columns for output file
    pred_df = val_df_processed.loc[:, ModelBasedConfig.pred_cols]

    execution_time = time.time() - start_time
    print(f"Model Training Time: {execution_time} s\n")

    return pred_df.values.tolist()


def get_error_distribution(test_data_path: str, output_file_name: str):
    # Read Validation Processed Dataframe
    val_df_processed = pd.read_csv(test_data_path, usecols=["user_id", "business_id", "rating"])

    # Read final output file
    pred_df = pd.read_csv(output_file_name)
    pred_df["error"] = abs(val_df_processed["rating"] - pred_df["prediction"])

    bins = [-np.inf, 1, 2, 3, 4, np.inf]

    # Define the labels for error levels
    labels = [">=0 and <1:", ">=1 and <2:", ">=2 and <3:", ">=3 and <4:", ">=4:"]

    # Bin the absolute differences into error levels
    pred_df["Error Distribution:"] = pd.cut(pred_df["error"], bins=bins, labels=labels, right=False)

    # Count the occurrences in each error level
    error_distribution = pred_df["Error Distribution:"].value_counts().sort_index()
    print(error_distribution)


def main(folder_path: str, test_file_name: str, output_file_name: str):
    start_time = time.time()
    # Process YELP Reviews Dataset
    process_data(folder_path, test_file_name)

    # Train Model Based Recommendation System
    pred_data = train_model(Path.yelp_train_processed, Path.yelp_val_processed)

    # Save the predictions
    save_data(pred_data, output_file_name)

    # Get the error distribution
    get_error_distribution(Path.yelp_val_processed, output_file_name)

    execution_time = time.time() - start_time
    print(f"Total Execution Time: {execution_time} s\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)

    # Read input parameters
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    main(folder_path, test_file_name, output_file_name)

# main("CompetitionStudentData", "CompetitionStudentData/yelp_val.csv", "output/output-final.csv")
