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

    def run(self, spark, train_file_name, test_file_name):
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

        return val_data


class ModelBased:
    def __init__(self):
        pass

    @staticmethod
    def read_csv_spark(path, sc):
        rdd = sc.textFile(path)
        header = rdd.first()
        rdd = rdd.filter(lambda row: row != header).map(lambda row: row.split(","))
        return rdd

    @staticmethod
    def read_json_spark(path, sc):
        return sc.textFile(path).map(lambda row: json.loads(row))

    @staticmethod
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

    @staticmethod
    def process_user(usr_rdd):
        usr_rdd = usr_rdd.map(
            lambda row: (row["user_id"], (float(row["average_stars"]), float(row["review_count"]), float(row["fans"])))
        ).cache()
        return usr_rdd.collectAsMap()

    @staticmethod
    def process_bus(bus_rdd):
        bus_rdd = bus_rdd.map(
            lambda row: (row["business_id"], (float(row["stars"]), float(row["review_count"])))
        ).cache()
        return bus_rdd.collectAsMap()

    @staticmethod
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

    def run(self, spark, folder_path, test_file_name):
        # Read and process the train data
        train_rdd = ModelBased.read_csv_spark(folder_path + "/yelp_train.csv", spark)

        review_rdd = ModelBased.read_json_spark(folder_path + "/review_train.json", spark)
        review_rdd = ModelBased.process_reviews(review_rdd)

        usr_rdd = ModelBased.read_json_spark(folder_path + "/user.json", spark)
        usr_rdd = ModelBased.process_user(usr_rdd)

        bus_rdd = ModelBased.read_json_spark(folder_path + "/business.json", spark)
        bus_rdd = ModelBased.process_bus(bus_rdd)

        # Read and process validation dataset
        val_rdd = ModelBased.read_csv_spark(test_file_name, spark).cache()

        # Train X and Y
        train_rdd = train_rdd.map(lambda x: ModelBased.process_train_data(x, review_rdd, usr_rdd, bus_rdd))

        # Valid x and Y
        val_processed = val_rdd.map(lambda x: ModelBased.process_train_data(x, review_rdd, usr_rdd, bus_rdd))

        # Extract X_train and Y_train
        X_train = train_rdd.map(lambda x: x[0]).cache()
        X_train = np.array(X_train.collect(), dtype="float32")
        Y_train = train_rdd.map(lambda x: x[1]).cache()
        Y_train = np.array(Y_train.collect(), dtype="float32")

        # Extract X_train and Y_train
        X_val = val_processed.map(lambda x: x[0]).cache()
        X_val = np.array(X_val.collect(), dtype="float32")

        params = {
            'lambda': 9.92724463758443,
            'alpha': 0.2765119705933928,
            'colsample_bytree': 0.5,
            'subsample': 0.8,
            'learning_rate': 0.02,
            'max_depth': 17,
            'random_state': 2020,
            'min_child_weight': 101,
            'n_estimators': 300,
        }

        xgb = XGBRegressor(**params)
        xgb.fit(X_train, Y_train)
        Y_pred = xgb.predict(X_val)

        pred_data = []
        for i, row in enumerate(val_rdd.collect()):
            pred_data.append([row[0], row[1], Y_pred[i]])

        return spark.parallelize(pred_data)


def hybrid_pred(preds, factor=0.5):
    wieghted_pred = factor * preds[0] + (1 - factor) * preds[1]
    return wieghted_pred


def task2_3(folder_path, test_file_name, output_file_name):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 2.3: Hybrid recommendation system")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        # Train the item-based collaborative recommendation system
        item_based = ItemBasedCF()
        item_based_pred = item_based.run(
            spark=spark, train_file_name=f"{folder_path}/yelp_train.csv", test_file_name=test_file_name
        )
        item_based_pred = item_based_pred.map(lambda x: ((x[0], x[1]), x[2])).persist()

        # Train the item-based collaborative recommendation system
        model_based = ModelBased()
        model_based_pred = model_based.run(spark=spark, folder_path=folder_path, test_file_name=test_file_name)
        model_based_pred = model_based_pred.map(lambda x: ((x[0], x[1]), x[2])).persist()

        FACTOR = 0.05222

        joined_preds = (
            item_based_pred.join(model_based_pred)
            .map(lambda x: [x[0][0], x[0][1], hybrid_pred(x[1], factor=FACTOR)])
            .cache()
        )

        save_data(joined_preds.collect(), output_file_name)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_1.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)

    # Read input parameters
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    task2_3(folder_path, test_file_name, output_file_name)

# task2_3("HW3StudentData", "HW3StudentData/yelp_val.csv", "t2_3.csv")
