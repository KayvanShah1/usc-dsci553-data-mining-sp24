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


class ItemBasedCF:
    def __init__(self):
        pass

    @staticmethod
    def prepare_dataset(data, split="train"):
        # Remove the header
        header = data.first()
        data = (
            data.filter(lambda row: row != header)
            .map(lambda row: row.split(","))
            .map(lambda row: (row[0], row[1], row[2]) if split == "train" else (row[0], row[1]))
        )
        return data

    @staticmethod
    def get_bus_to_usr_map(train_data):
        # Group by business_id and collect the corresponding set of users
        bus2user = (
            train_data.map(lambda x: (x[1], (x[0], float(x[2]))))
            .groupByKey()
            .mapValues(lambda vals: {"users": dict(vals), "avg_rating": sum(val[1] for val in vals) / len(vals)})
        )
        return bus2user.collectAsMap()

    @staticmethod
    def get_usr_to_bus_map(train_data):
        # Group by user_id and collect the corresponding set of businesses
        user2bus = (
            train_data.map(lambda x: (x[0], (x[1], float(x[2]))))
            .groupByKey()
            .mapValues(lambda vals: {"business": dict(vals)})
        )
        return user2bus.collectAsMap()

    @staticmethod
    def compute_pearson_similarity(data, item2user_dict):
        """
        Formala: r = Σᵢ((xᵢ − mean(x))(yᵢ − mean(y))) (√Σᵢ(xᵢ − mean(x))² √Σᵢ(yᵢ − mean(y))²)⁻¹
        """
        # Unpack the data
        item1, item2 = data

        # Find common user to calculate co-rated averages
        users_item1 = set(item2user_dict[item1]["users"].keys())
        users_item2 = set(item2user_dict[item2]["users"].keys())
        common_users = users_item1.intersection(users_item2)

        if len(common_users) <= 1:
            similarity = (5 - abs(item2user_dict[item1]["avg_rating"] - item2user_dict[item2]["avg_rating"])) / 5
        else:
            r1 = []
            r2 = []
            # Get ratings of common users for both business
            for usr in common_users:
                r1.append(item2user_dict[item1]["users"][usr])
                r2.append(item2user_dict[item2]["users"][usr])

            # Center the ratings by subtracting the co-rated average rating
            r1_bar = sum(r1) / len(r1)
            r2_bar = sum(r2) / len(r2)
            r1 = [r - r1_bar for r in r1]
            r2 = [r - r2_bar for r in r2]

            # Compute weight for the item pair
            numer = sum([a * b for a, b in zip(r1, r2)])
            denom = ((sum([a**2 for a in r1])) ** 0.5) * (sum([b**2 for b in r2]) ** 0.5)

            similarity = 0 if denom == 0 else numer / denom

        return similarity

    @staticmethod
    def predict_rating(data, bus2user_dict, user2bus_dict, neighbours=15):
        """Perform Item-based Collaborative filtering on prepared data."""
        # Unpack the data
        user, business = data

        # Return avg rating if user or business is not present in the dataset
        if user not in user2bus_dict or business not in bus2user_dict:
            return 3.0

        # Pearson similarities for rating prediction
        pc = []

        for item in user2bus_dict[user]["business"].keys():
            # Compute pearson similarity for each business pair
            similarity = ItemBasedCF.compute_pearson_similarity((business, item), bus2user_dict)
            pc.append((similarity, bus2user_dict[item]["users"][user]))

        # Calculate the predicted rating
        top_pc = sorted(pc, key=lambda x: -x[0])[:neighbours]
        x, y = 0, 0
        for p, r in top_pc:
            x += p * r
            y += abs(p)
        predicted_rating = 3.5 if y == 0 else x / y

        return predicted_rating

    def run(train_file_name, test_file_name):
        # Initialize Spark
        conf = SparkConf().setAppName("Task 2.1: Item-Based Collaborative Filtering")
        spark = SparkContext(conf=conf).getOrCreate()
        spark.setLogLevel("ERROR")

        try:
            start_time = time.time()

            # Read and process the train data
            train_data = spark.textFile(train_file_name)
            train_data = ItemBasedCF.prepare_dataset(train_data, split="train")

            # Preprocess train data to get mapping dictionaries
            bus2user_dict = ItemBasedCF.get_bus_to_usr_map(train_data)
            user2bus_dict = ItemBasedCF.get_usr_to_bus_map(train_data)

            # Read and prepare validation data
            val_data = spark.textFile(test_file_name)
            val_data = ItemBasedCF.prepare_dataset(val_data, split="valid").cache()

            val_data = val_data.map(
                lambda x: [x[0], x[1], ItemBasedCF.predict_rating(x, bus2user_dict, user2bus_dict)]
            ).cache()

            execution_time = time.time() - start_time
            print(f"Duration: {execution_time}\n")

            return val_data

        finally:
            # Stop Spark
            spark.stop()
