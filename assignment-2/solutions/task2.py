import sys
import time

from pyspark import SparkConf, SparkContext
from task1 import SON


def generate_baskets(data, filter_threshold):
    """
    Generate baskets based on the specified case.
    """
    baskets = (
        data.map(lambda x: (x[0], [x[1]]))
        .reduceByKey(lambda a, b: a + b)
        .filter(lambda x: len(x[1]) > filter_threshold)
    )
    return baskets


def create_date_customer_id(date, customer_id):
    date = date.replace('"', "").split("/")
    date = f"{date[0]}/{date[1]}/{date[2][2:]}"
    customer_id = int(customer_id.replace('"', ""))
    return f"{date}-{customer_id}"


def process_prod_id(prod_id):
    prod_id = int(prod_id.replace('"', ""))
    return f"{prod_id}"


def preprocess_data(data):
    # Drop the header row
    data_header = data.first()
    data = (
        data.filter(lambda row: row != data_header)
        .map(lambda row: row.split(","))
        .map(lambda row: [create_date_customer_id(row[0], row[1]), process_prod_id(row[5])])
    )
    return data


def task2(filter_threshold, support, input_file_path, output_file_path):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 2")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        # Read the input data
        data = spark.textFile(input_file_path)
        data = preprocess_data(data)

        # Generate baskets based on the specified filter threshold
        baskets = generate_baskets(data, filter_threshold).cache()

        # Run the SON Algorithm
        candidates, frequent_itemsets = SON(baskets, support, num_buckets=1000)

        # Output the results
        with open(output_file_path, "w") as f:
            f.write("Candidates:\n")
            for _, itemsets in candidates.items():
                itemsets = ",".join(map(str, itemsets)).replace(",)", ")")
                f.write(f"{itemsets}\n\n")
            f.write("Frequent Itemsets:\n")
            for line, itemsets in frequent_itemsets.items():
                itemsets = ",".join(map(str, itemsets)).replace(",)", ")")
                f.write(f"{itemsets}")
                if line < len(frequent_itemsets):
                    f.write("\n\n")

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: spark-submit --executor-memory 4G --driver-memory 4G "
            "task2.py <filter_threshold> <support> <input_file_path> <output_file_path>"
        )
        sys.exit(1)

    # Read input parameters
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    task2(filter_threshold, support, input_file_path, output_file_path)
