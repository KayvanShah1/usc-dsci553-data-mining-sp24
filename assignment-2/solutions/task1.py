import sys
import time
from pyspark import SparkContext, SparkConf


def generate_baskets(data, case):
    """
    Generate baskets based on the specified case.
    """
    baskets = (
        data.map(lambda line: line.split(",")).map(lambda x: (x[0], [x[1]]))
        if case == 1
        else data.map(lambda line: line.split(",")).map(lambda x: (x[1], [x[0]]))
    )
    return baskets.reduceByKey(lambda a, b: a + b)


def hash_function(item, num_buckets):
    """
    Simple hash function to distribute items into different buckets.
    """
    return hash(item) % num_buckets


def count_item_pairs(chunk, num_buckets):
    """
    Count item pairs using PCY approach.
    """
    pair_counts = {}
    for basket in chunk:
        items = basket[1]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                item1, item2 = items[i], items[j]
                bucket_index = hash_function((item1, item2), num_buckets)
                pair_counts[bucket_index] = pair_counts.get(bucket_index, {})
                pair_counts[bucket_index][(item1, item2)] = (
                    pair_counts[bucket_index].get((item1, item2), 0) + 1
                )
    return pair_counts


def filter_candidates(pair_counts, support):
    """
    Filter candidate pairs based on support threshold.
    """
    candidates = set()
    for bucket in pair_counts.values():
        for pair, count in bucket.items():
            if count >= support:
                candidates.add(pair)
    return candidates


def generate_frequent_itemsets(chunk, candidates):
    """
    Generate frequent itemsets within each chunk.
    """
    frequent_itemsets = []
    for basket in chunk:
        items = set(basket[1])
        for candidate in candidates:
            if all(item in items for item in candidate):
                frequent_itemsets.append((candidate, 1))
    return frequent_itemsets


def run_PCY_algorithm(data, case_number, support, num_buckets, sc):
    """
    Run the PCY algorithm.
    """
    # Generate baskets based on the specified case
    baskets = generate_baskets(data, case_number)

    # Implement the PCY Algorithm
    # Split the data into chunks (if needed)
    chunks = baskets.randomSplit([1] * baskets.count(), seed=1)

    # First pass: Count item pairs
    pair_counts = (
        sc.parallelize(chunks)
        .flatMap(lambda chunk: count_item_pairs(chunk.collect(), num_buckets))
        .reduceByKey(lambda a, b: {**a, **b})
        .collectAsMap()
    )

    # Filter candidates based on support threshold
    candidates = filter_candidates(pair_counts, support)

    # Second pass: Generate frequent itemsets
    frequent_itemsets = []
    for chunk in chunks:
        frequent_itemsets += generate_frequent_itemsets(chunk.collect(), candidates, case_number)

    return frequent_itemsets


def task1(case_number, support, input_file_path, output_file_path):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 1")
    sc = SparkContext(conf=conf)

    # Read the input data
    data = sc.textFile(input_file_path)

    try:
        # Run the PCY Algorithm
        start_time = time.time()
        frequent_itemsets = run_PCY_algorithm(data, case_number, support, num_buckets=1000, sc=sc)
        execution_time = time.time() - start_time

        # Output the results
        with open(output_file_path, "w") as f:
            f.write(f"Duration: {execution_time}\n")
            f.write("Candidates:\n")
            f.write("Frequent Itemsets:\n")
            for itemset in frequent_itemsets:
                f.write(f"{itemset}\n\n")

    finally:
        # Stop Spark
        sc.stop()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Usage: spark-submit --executor-memory 4G --driver-memory 4G "
            "task1.py <case_number> <support> <input_file_path> <output_file_path>"
            " --conf spark.driver.extraJavaOptions=-Dlog4j.configuration=log4j.properties"
        )
        sys.exit(1)

    # Read input parameters
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    task1(case_number, support, input_file_path, output_file_path)
