# %%writefile task1.py
import os
import sys
import time

from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import array_intersect, col, size

# os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"


def read_csv(spark: SparkSession, file_path: str):
    """
    Read CSV file into DataFrame
    """
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df


def construct_graph(spark: SparkSession, df, filter_threshold: int):
    """
    Function to construct the social network graph.
    """
    # Create user to business map
    user2bus = (
        df.groupBy("user_id")
        .agg({"business_id": "collect_set"})
        .withColumnRenamed("collect_set(business_id)", "business_set")
    )

    # Create edge dataframe
    edges = (
        user2bus.alias("u1")
        .join(user2bus.alias("u2"), col("u1.user_id") != col("u2.user_id"))
        # Filter edges by threshold
        .where(size(array_intersect(col("u1.business_set"), col("u2.business_set"))) >= filter_threshold)
        .select(col("u1.user_id").alias("src"), col("u2.user_id").alias("dst"))
        .distinct()
    )

    # Create nodes DataFrame
    nodes_src = edges.select(col("src").alias("id"))
    nodes_dst = edges.select(col("dst").alias("id"))
    nodes = nodes_src.union(nodes_dst).distinct()

    # Construct GraphFrame
    graph = GraphFrame(nodes, edges)

    return graph


def detect_communities(graph):
    """
    Function to detect communities using Label Propagation Algorithm.
    """
    # Use the Label Propagation Algorithm to detect communities
    communities = graph.labelPropagation(maxIter=5)
    return communities


def save_communities(communities, output_file_path):
    """
    Function to save communities to a txt file.
    """
    # Sort communities by size and then by lexicographical order
    sorted_communities = (
        communities.rdd.map(lambda x: (x[1], x[0]))
        .groupByKey()
        .map(lambda x: sorted(list(x[1])))
        .sortBy(lambda x: (len(x), x))
        .collect()
    )

    # Save communities to the output file
    with open(output_file_path, "w") as file:
        for community in sorted_communities:
            file.write(", ".join(community) + "\n")


def task1(filter_threshold: int, input_file_path: str, community_output_file_path: str):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 1").set("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    sc = SparkContext(conf=conf).getOrCreate()
    spark = SparkSession(sparkContext=sc)
    spark.sparkContext.setLogLevel("ERROR")

    try:
        start_time = time.time()

        df = read_csv(spark, input_file_path)

        # Construct graph
        graph = construct_graph(spark, df, filter_threshold)

        # Detect Communities
        communities = detect_communities(graph)

        # Save the generated communities data
        save_communities(communities, community_output_file_path)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 4:
        print(
            "Usage: spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py <filter threshold>"
            " <input file path> <community output file path>"
        )
        sys.exit(1)

    # Parse command-line arguments
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    community_output_file_path = sys.argv[3]

    # Call task1 function
    task1(filter_threshold, input_file_path, community_output_file_path)

# task1(2, Path.input_csv_file, Path.task1_output)
