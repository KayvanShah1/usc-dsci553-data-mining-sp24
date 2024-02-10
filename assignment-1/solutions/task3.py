import json
import sys
from time import time

from pyspark import SparkContext


def toCSVLine(item):
    item = ",".join(str(element) for element in item)
    return f"{item}\n"


def task3(review_file_path, business_file_path, output_file_path_qa, output_file_path_qb):
    spark = SparkContext(appName="Yelp Data Exploration (Task 3)").getOrCreate()

    review_rdd = spark.textFile(review_file_path).map(json.loads).cache()
    business_rdd = spark.textFile(business_file_path).map(json.loads).cache()

    result_qb = {"m1": 0, "m2": 0, "reason": ""}

    try:
        # Average stars for each city
        lcca_start_time = time()

        review_rdd = review_rdd.map(lambda x: (x["business_id"], x["stars"]))
        business_rdd = business_rdd.map(lambda x: (x["business_id"], x["city"]))

        merged_rdd = business_rdd.join(review_rdd).map(lambda x: x[1])

        agg_rdd = merged_rdd.groupByKey().map(lambda x: (x[0], sum(x[1]) / len(x[1])))
        lcca_end_time = time()
        # lcca_time = loading time + time to create and collect averages
        lcca_time = lcca_end_time - lcca_start_time

        spark_sorting_start_time = time()
        sorted_rdd = agg_rdd.sortBy(lambda x: (-x[1], x[0]), ascending=True)
        top_10_cities = sorted_rdd.take(10)
        print(top_10_cities)
        spark_sorting_end_time = time()

        spark_sorting_time = spark_sorting_end_time - spark_sorting_start_time

        with open(output_file_path_qa, "w") as f:
            f.write("city,stars\n")
            f.writelines([toCSVLine(i) for i in sorted_rdd.collect()])

        # Performance Comparsion with Python based sorting
        py_sorting_start_time = time()
        sorted_list = sorted(agg_rdd.collect(), key=lambda x: (-x[1], x[0]))
        top_10_cities = sorted_list[:10]
        print(top_10_cities)
        py_sorting_end_time = time()

        py_sorting_time = py_sorting_end_time - py_sorting_start_time

        result_qb["m1"] = lcca_time + py_sorting_time
        result_qb["m2"] = lcca_time + spark_sorting_time
        reason = (
            "Method 1 (M1): Sorting is done in Python after collecting data from RDD. This"
            "may be faster for smaller datasets but can be slower for larger datasets due to the"
            "overhead of collecting data to the driver.\n\n"
            "Method 2 (M2): Sorting is done in Spark without collecting data to the driver. This"
            "approach can be more scalable for larger datasets but may incur more overhead due to"
            "Spark's internal operations."
        )
        result_qb["reason"] = reason

        with open(output_file_path_qb, "w") as f:
            json.dump(result_qb, f, indent=4)

    finally:
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: spark-submit --executor-memory 4G --driver-memory 4G"
            " task2.py <review_file_path> <business_file_path> <output_file_path_qa>"
            " <output_file_path_qb>"
        )
        sys.exit(1)

    review_file_path = sys.argv[1]
    business_file_path = sys.argv[2]
    output_file_path_qa = sys.argv[3]
    output_file_path_qb = sys.argv[4]

    task3(review_file_path, business_file_path, output_file_path_qa, output_file_path_qb)
