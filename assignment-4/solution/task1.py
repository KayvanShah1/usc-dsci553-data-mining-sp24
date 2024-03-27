import time

from pyspark import SparkConf, SparkContext


def task1():
    # Initialize Spark
    conf = SparkConf().setAppName("Task 1").set("spark.executor.memory", "4G").set("spark.driver.memory", "4G")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    task1()
