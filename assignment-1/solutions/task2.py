import json
import sys
from time import time

from pyspark import SparkContext


def custom_partition_func(record, n_partition):
    return hash(record[0]) % n_partition


def task2(review_file_path, output_file_path, n_partition):
    spark = SparkContext(appName="Yelp Data Exploration (Task 2)").getOrCreate()

    review_rdd = spark.textFile(review_file_path).map(json.loads).cache()

    result = {
        "default": {"n_partition": 0, "n_items": [], "exe_time": 0},
        "customized": {"n_partition": 0, "n_items": [], "exe_time": 0},
    }

    try:
        # Default partition function
        start = time()
        _ = (
            review_rdd.map(lambda x: (x["business_id"], 1))
            .reduceByKey(lambda a, b: a + b)
            .sortBy(lambda x: (-x[1], x[0]), ascending=True)
            .take(10)
        )
        end = time()

        result["default"]["n_partition"] = review_rdd.getNumPartitions()

        result["default"]["n_items"] = review_rdd.glom().map(len).collect()

        result["default"]["exe_time"] = end - start

        # Customized partition function
        start = time()
        partitioned_rdd = (
            review_rdd.map(lambda x: (x["business_id"], 1))
            .partitionBy(n_partition, lambda record: custom_partition_func(record, n_partition))
            .cache()
        )

        _ = (
            partitioned_rdd.reduceByKey(lambda a, b: a + b)
            .sortBy(lambda x: (-x[1], x[0]), ascending=True)
            .take(10)
        )
        end = time()

        result["customized"]["n_partition"] = partitioned_rdd.getNumPartitions()

        result["customized"]["n_items"] = partitioned_rdd.glom().map(len).collect()

        result["customized"]["exe_time"] = end - start

        with open(output_file_path, "w") as f:
            json.dump(result, f, indent=4)

    finally:
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: spark-submit --executor-memory 4G --driver-memory 4G"
            " task2.py <review_file_path> <output_file_path> <n_partitions>"
        )
        sys.exit(1)

    review_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    n_partition = int(sys.argv[3])

    task2(review_file_path, output_file_path, n_partition)
