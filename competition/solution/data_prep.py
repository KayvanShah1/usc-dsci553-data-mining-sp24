import csv
import json
import os
from datetime import datetime

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark import SparkContext

# Download the VADER lexicon if not already downloaded
nltk.download("vader_lexicon", quiet=True)


class DataReader:
    def __init__(self, sc: SparkContext, folder_path: str):
        self.sc = sc
        self.folder_path = folder_path

    def get_path(self, filename: str):
        return os.path.join(self.folder_path, filename)

    def read_csv_spark(self, path: str):
        path = self.get_path(path)
        rdd = self.sc.textFile(path)
        header = rdd.first()
        rdd = rdd.filter(lambda row: row != header).map(lambda row: row.split(","))
        return rdd, header.split()

    def read_json_spark(self, path: str):
        path = self.get_path(path)
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
        row["latitude"] = float(row["latitude"])
        row["longitude"] = float(row["longitude"])
        row["stars"] = float(row["stars"])

        # Delete keys
        row = {k: v for k, v in row.items() if k not in BusinessData.keys_to_delete}
        return row

    @staticmethod
    def generate_mapping(bus_rdd):
        # Extract unique values for state and city
        state_to_index = bus_rdd.map(lambda row: row["state"]).distinct().zipWithIndex().collectAsMap()
        city_to_index = bus_rdd.map(lambda row: row["city"]).distinct().zipWithIndex().collectAsMap()

        return state_to_index, city_to_index

    def process(bus_rdd):
        bus_rdd = bus_rdd.map(lambda row: BusinessData.parse_row(row)).map(
            lambda row: (
                row["business_id"],
                (
                    row["stars"],
                    row["review_count"],
                    row["is_open"],
                    row["num_attrs"],
                    row["num_categories"],
                    row["latitude"],
                    row["longitude"],
                ),
            )
        )
        return bus_rdd


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
        row = {k: v for k, v in row.items() if k not in UserData.keys_to_delete}
        return row

    def process(user_rdd):
        user_rdd = user_rdd.map(lambda row: UserData.parse_row(row)).map(
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
        return user_rdd


class ReviewData:
    keys_to_delete = ["review_id", "date", "text"]

    @staticmethod
    def parse_row(row):

        # Delete keys
        row = {k: v for k, v in row.items() if k not in ReviewData.keys_to_delete}
        return row

    def process(rdd):
        rdd = (
            rdd.map(lambda row: ReviewData.parse_row(row))
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
        return rdd


class TipData:
    keys_to_delete = ["date", "text"]

    @staticmethod
    def parse_row(row):
        # Delete keys
        row = {k: v for k, v in row.items() if k not in TipData.keys_to_delete}
        return row

    def process(rdd):
        rdd = (
            rdd.map(lambda row: TipData.parse_row(row))
            # Calculate sum of likes and review count by user
            .map(lambda row: ((row["user_id"], row["business_id"]), (row["likes"], 1))).reduceByKey(
                lambda x, y: (x[0] + y[0], x[1] + y[1])
            )
        )
        return rdd


def calculate_sentiment(text):
    # Initialize the VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Perform sentiment analysis
    sentiment_scores = sia.polarity_scores(text)

    # Return the compound sentiment score
    return sentiment_scores["compound"]


def save_data(data: list, output_file_name: str):
    header = ["user_id", "business_id", "prediction"]
    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
