import json
import sys
from datetime import datetime

from pyspark import SparkContext


def task1(review_file_path, output_file_path):
    spark = SparkContext(appName="Yelp Data Exploration (Task 1)").getOrCreate()

    review_rdd = spark.textFile(review_file_path).map(json.loads).cache()

    result = {
        "n_review": 0,
        "n_review_2018": 0,
        "n_user": 0,
        "top10_user": [],
        "n_business": 0,
        "top10_business": [],
    }

    try:
        result["n_review"] = review_rdd.count()

        result["n_review_2018"] = review_rdd.filter(
            lambda x: datetime.strptime(x["date"], "%Y-%m-%d %H:%M:%S").year == 2018
        ).count()

        result["n_user"] = review_rdd.map(lambda x: x["user_id"]).distinct().count()

        result["top10_user"] = (
            review_rdd.map(lambda x: (x["user_id"], 1))
            .reduceByKey(lambda a, b: a + b)
            .sortBy(lambda x: (-x[1], x[0]), ascending=True)
            .take(10)
        )

        result["n_business"] = review_rdd.map(lambda x: x["business_id"]).distinct().count()

        result["top10_business"] = (
            review_rdd.map(lambda x: (x["business_id"], 1))
            .reduceByKey(lambda a, b: a + b)
            .sortBy(lambda x: (-x[1], x[0]), ascending=True)
            .take(10)
        )

        with open(output_file_path, "w") as f:
            json.dump(result, f, indent=4)

    finally:
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: spark-submit --executor-memory 4G --driver-memory 4G "
            "task1.py <review_file_path> <output_file_path>"
            " --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j.properties"
        )
        sys.exit(1)

    review_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    task1(review_file_path, output_file_path)
