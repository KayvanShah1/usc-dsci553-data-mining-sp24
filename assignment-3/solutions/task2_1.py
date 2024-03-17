import csv
import math
import statistics
import sys
import time

from pyspark import SparkConf, SparkContext


def prepare_dataset(data, split="train"):
    # Remove the header
    header = data.first()
    data = (
        data.filter(lambda row: row != header)
        .map(lambda row: row.split(","))
        .map(lambda row: (row[0], row[1], row[2]) if split == "train" else (row[0], row[1]))
    )
    return data


def preprocess_train_data(train_data):
    # Group by business_id and collect the corresponding set of users
    bus2user = train_data.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)
    bus2user_dict = bus2user.collectAsMap()

    # Group by user_id and collect the corresponding set of businesses
    user2bus = train_data.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)
    user2bus_dict = user2bus.collectAsMap()

    # Group by business_id and collect the corresponding set of users with ratings
    bus2user_rating = train_data.map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict)
    bus2user_rating_dict = bus2user_rating.collectAsMap()

    # Calculate average rating for each business
    bus_avg = (
        train_data.map(lambda row: (row[1], float(row[2])))
        .groupByKey()
        .mapValues(lambda ratings: sum(ratings) / len(ratings))
        .collectAsMap()
    )

    # Calculate average rating for each user
    user_avg = (
        train_data.map(lambda row: (row[0], float(row[2])))
        .groupByKey()
        .mapValues(lambda ratings: sum(ratings) / len(ratings))
        .collectAsMap()
    )

    return bus2user_dict, user2bus_dict, bus2user_rating_dict, bus_avg, user_avg


def compute_pearson_similarity(data, item2user_dict, item2user_rating_dict):
    """
    Formala: r = Σᵢ((xᵢ − mean(x))(yᵢ − mean(y))) (√Σᵢ(xᵢ − mean(x))² √Σᵢ(yᵢ − mean(y))²)⁻¹
    """
    # Unpack the data
    item1, item2 = data

    # Find common user to calculate co-rated averages
    common_users = item2user_dict[item1].intersection(item2user_dict[item2])

    # Get ratings of common users for both business
    r1 = [item2user_rating_dict[item1][usr] for usr in common_users]
    r2 = [item2user_rating_dict[item2][usr] for usr in common_users]

    # Center the ratings by subtracting the co-rated average rating
    r1 = [r - statistics.mean(r1) for r in r1]
    r2 = [r - statistics.mean(r2) for r in r2]

    # Compute weight for the item pair
    numer = sum([a * b for a, b in zip(r1, r2)])
    denom = math.sqrt(sum([math.pow(a, 2) for a in r1])) * math.sqrt(sum([math.pow(b, 2) for b in r2]))

    similarity = 0 if denom == 0 else numer / denom

    return similarity


def predict_rating(data, bus2user_dict, user2bus_dict, bus2user_rating_dict, bus_avg, user_avg):
    """Perform Item-based Collaborative filtering on prepared data."""
    # Unpack the data
    user, business = data

    # Return avg rating if user or business is not present in the dataset
    if user not in user2bus_dict.keys():
        return 3.5
    if business not in bus2user_dict:
        return user_avg[user]

    # Pearson similarities for rating prediction
    pc = []

    for item in user2bus_dict[user]:
        # Compute pearson similarity for each business pair
        similarity = compute_pearson_similarity((business, item), bus2user_dict, bus2user_rating_dict)

        pc.append((similarity, bus2user_rating_dict[item][user]))

    # Calculate the predicted rating
    top_pc = sorted(pc, key=lambda x: -x[0])[:15]
    x, y = 0, 0
    for p, r in top_pc:
        x += p * r
        y += abs(p)
    predicted_rating = 3.5 if y == 0 else x / y

    return predicted_rating


def task2_1(train_file_name, test_file_name, output_file_name):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 2.1: Item-Based Collaborative Filtering")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        # Read and process the train data
        train_data = spark.textFile(train_file_name)
        train_data = prepare_dataset(train_data, split="train")
        (bus2user_dict, user2bus_dict, bus2user_rating_dict, bus_avg, user_avg) = preprocess_train_data(train_data)

        # Read and prepare validation data
        val_data = spark.textFile(test_file_name)
        val_data = prepare_dataset(val_data, split="valid")

        # Predict ratings for validation dataset
        val_data = (
            val_data.map(
                lambda x: (
                    x[0],
                    x[1],
                    predict_rating(x, bus2user_dict, user2bus_dict, bus2user_rating_dict, bus_avg, user_avg),
                )
            )
            .map(lambda x: list(x))
            .collect()
        )

        header = ["user_id", "business_id", "prediction"]
        with open(output_file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(val_data)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>")
        sys.exit(1)

    # Read input parameters
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    task2_1(train_file_name, test_file_name, output_file_name)

# task2_1("HW3StudentData/yelp_train.csv", "HW3StudentData/yelp_val.csv", "t2_1.csv")
