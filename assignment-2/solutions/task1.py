import sys
import time
from pyspark import SparkContext, SparkConf

from collections import Counter
from itertools import combinations, chain


def generate_baskets(data, case):
    """
    Generate baskets based on the specified case.
    """
    if case == 1:
        # Group Business
        baskets = data.map(lambda line: line.split(",")).map(lambda x: (x[0], [x[1]]))
    elif case == 2:
        # Group User
        baskets = data.map(lambda line: line.split(",")).map(lambda x: (x[1], [x[0]]))
    return baskets.reduceByKey(lambda a, b: a + b)


def hash_function(item1, item2, num_buckets):
    """
    Polynomial hash function to distribute items into different buckets.
    """
    hash_val = (int(item1) * 31) + int(item2)
    return hash_val % num_buckets


def PCY(baskets, total_num_baskets, support, num_buckets):
    """
    Run the PCY algorithm to find candidates
    """
    candidates = []
    baskets = list(baskets)

    # Calculate the support ratio based on the number of baskets in the current partition
    # and the total number of baskets
    support_ratio = len(baskets) / total_num_baskets

    # Calculate the partition support threshold by multiplying the support ratio with the desired support
    partition_support_threshold = support_ratio * support

    # First pass:
    # Initialize counters to keep track of item counts, pair counts, and hash counts
    item_counts = Counter()
    pair_counts = Counter()
    hash_counts = Counter()

    # Initialize a bitmap list with zeros, representing whether each hash bucket meets the support threshold
    bitmap = [0] * num_buckets

    for basket in baskets:
        # Count item pairs
        item_counts.update(basket)

        pairs = list(combinations(basket, 2))
        pair_counts.update(pairs)

        # Hash pairs and count the number of occurence every hash
        hashes = [hash_function(item[0], item[1], num_buckets) for item in pairs]
        hash_counts.update(hashes)

    # Update the bitmap if count exceeds support threshold
    for h, count in hash_counts.items():
        if count >= partition_support_threshold:
            bitmap[h] = 1

    # Second Pass:
    frequent_singles = []
    for item, count in item_counts.items():
        if count >= partition_support_threshold:
            frequent_singles.append(item)
            candidates.append((tuple([item]), 1))

    frequent_pairs = []
    for item in pair_counts.keys():
        hash_val = hash_function(item[0], item[1], num_buckets)
        if bitmap[hash_val] == 1:
            frequent_pairs.append(item)

    if len(frequent_pairs) == 0:
        return []

    # Filter items from basket that are not frequent
    baskets = [[item for item in basket if item in frequent_singles] for basket in baskets]

    # Initialize candidate dictionary
    candidate_dict = {pair: 0 for pair in frequent_pairs}

    # Count occurrences of frequent pairs in baskets
    for basket in baskets:
        for pair in frequent_pairs:
            if set(pair).issubset(basket):
                candidate_dict[pair] += 1

    # Filter the pair that do not meet the support threshold
    candidate_dict = {pair: count for pair, count in candidate_dict.items() if count >= partition_support_threshold}

    # Append new candidates to the candidates list
    candidates += [(pair, 1) for pair in candidate_dict.keys()]

    # Valid Candidates
    val_candidates = candidate_dict.keys()

    # Check for frequent pairs in larger sets
    k = 3
    while len(val_candidates) > 0:
        unq_singles = set(chain.from_iterable(val_candidates))
        itemsets = list(combinations(unq_singles, k))

        # Initialize candidate dictionary
        candidate_dict = {itemset: 0 for itemset in itemsets}

        # Count occurrences of frequent pairs in baskets
        for basket in baskets:
            for itemset in itemsets:
                # Check if the candidate itemset is a subset of the current basket
                if set(itemset).issubset(basket):
                    candidate_dict[itemset] += 1

        # Filter the pair that do not meet the support threshold
        candidate_dict = {
            itemset: count for itemset, count in candidate_dict.items() if count >= partition_support_threshold
        }

        # Append new candidates to the candidates list
        candidates += [(itemset, 1) for itemset in candidate_dict.keys()]

        # Update Valid Candidates
        val_candidates = candidate_dict.keys()

        # Update Order of larger itemset for next iteration
        k += 1

    return candidates


def check_itemsets_in_basket(baskets, candidates):
    """
    Check if candidate itemsets are frequent in the given baskets.

    Parameters:
        baskets (iterable): An iterable containing baskets of items.
        candidates (iterable): An iterable containing candidate itemsets.

    Returns:
        frequent_itemsets (list): A list of frequent itemsets found in the baskets.
    """
    candidates = list(candidates)
    frequent_itemsets = []

    for basket in baskets:
        for itemset in candidates:
            # Check if the candidate itemset is a subset of the current basket
            if set(itemset).issubset(basket):
                frequent_itemsets.append(itemset)

    return frequent_itemsets


def SON(baskets, support, num_buckets=1000):
    """
    Perform SON algorithm to find frequent itemsets.

    Args:
        baskets (RDD): RDD containing baskets of items.
        support (int): Minimum support threshold.
        num_buckets (int, optional): Number of buckets for the PCY algorithm. Defaults to 1000.

    Returns:
        tuple: A tuple containing two dictionaries:
            - candidates_map: A dictionary mapping length of itemsets to the list of frequent candidates.
            - frequent_itemsets: A dictionary mapping length of frequent itemsets to the list of frequent itemsets.
    """
    # Count the number of baskets
    num_baskets = baskets.count()

    # Extract individual items from baskets
    items = baskets.values()

    # Stage 1: Finding Frequent Itemsets using PCY
    candidates = (
        items.mapPartitions(lambda chunk: PCY(chunk, num_baskets, support, num_buckets))
        .reduceByKey(lambda x, y: x + y)
        .map(lambda x: tuple(sorted(x[0])))
        .distinct()
        .sortBy(lambda x: (len(x), x))
    )

    # Group candidates by length and collect as a map
    candidates_map = (
        candidates.groupBy(lambda x: len(x))
        .mapValues(list)  # Convert grouped values to a list
        .sortByKey()
        .collectAsMap()
    )

    # Retrieve frequent candidates as a list
    candidates = candidates.collect()

    # Stage 2: Candidate Pruning
    frequent_itemsets = (
        items.mapPartitions(lambda chunk: check_itemsets_in_basket(chunk, candidates))
        .map(lambda itemset: (itemset, 1))
        .reduceByKey(lambda x, y: x + y)
        .filter(lambda x: x[1] >= support)
        .map(lambda x: tuple(sorted(x[0])))
        .sortBy(lambda x: (len(x), x))
        .groupBy(lambda x: len(x))
        .mapValues(list)  # Convert grouped values to a list
        .sortByKey()
        .collectAsMap()
    )
    return candidates_map, frequent_itemsets


def task1(case_number, support, input_file_path, output_file_path):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 1")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()
        # Read the input data
        data = spark.textFile(input_file_path)

        # Drop the header row
        data = data.filter(lambda row: row != "user_id,business_id")

        # Generate baskets based on the specified case
        baskets = generate_baskets(data, case_number).cache()

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
            "task1.py <case_number> <support> <input_file_path> <output_file_path>"
        )
        sys.exit(1)

    # Read input parameters
    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    task1(case_number, support, input_file_path, output_file_path)
