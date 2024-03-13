import csv
import random
import sys
import time
from itertools import combinations

from pyspark import SparkConf, SparkContext

random.seed(37)

# Define constants
NUM_HASH_FUNCTIONS = 50
PRIME_NUMBER = 15485863
ROWS_PER_BAND = 2
BANDS = NUM_HASH_FUNCTIONS // ROWS_PER_BAND


def prepare_dataset(data):
    # Remove the header
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(","))

    # Find unique users and map it to an index
    usr_to_idx = data.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()

    # Group users that has reviewed a business
    business_user = data.map(lambda row: (row[1], [row[0]])).reduceByKey(lambda a, b: a + b)
    return business_user, usr_to_idx


def generate_hash_function_params(max_range, count):
    """Generate random hash function parameters within a specified range."""
    hash_funcs = []
    for _ in range(count):
        a = random.randint(1, max_range)  # Random coefficient 'a'
        b = random.randint(0, max_range)  # Random intercept 'b'
        hash_funcs.append((a, b))
    return hash_funcs


def hash_item(item, params, num_bins):
    """Hash an item using given hash function parameters.
    Calculate hash value using the formula: ((a * item + b) % PRIME_NUMBER) % num_bins
    """
    hash_val = ((params[0] * item + params[1]) % PRIME_NUMBER) % num_bins
    return hash_val


def build_minhash_signature_matrix(hash_funcs, users, num_bins):
    """Build the minhash signature matrix for a set of users."""
    mhs = []
    for params in hash_funcs:
        minhash = float("inf")
        for user in users:
            # Hash each user and find the minimum hash value
            hash_val = hash_item(user, params, num_bins)
            minhash = min(minhash, hash_val)
        mhs.append(minhash)
    return mhs


def jaccard_similarity(pair, bus_user_dict):
    """
    Calculate Jaccard similarity for a candidate pair of businesses.

    Args:
        pair (tuple): A pair of business IDs.
        bus_user_dict (dict): Dictionary mapping business IDs to sets of user IDs.

    Returns:
        tuple: A tuple containing the business pair and their Jaccard similarity.
    """
    # Extract business IDs from the pair
    bus1, bus2 = pair

    # Get sets of users who reviewed each business
    user1 = set(bus_user_dict[bus1])
    user2 = set(bus_user_dict[bus2])

    # Calculate Jaccard similarity
    intersection = len(user1 & user2)
    union = len(user1 | user2)
    similarity = intersection / union if union != 0 else 0

    return (bus1, bus2), similarity


def jaccard_based_lsh(prepared_data):
    """Perform Jaccard-based Locality Sensitive Hashing (LSH) on prepared data.

    This function applies LSH to find candidate pairs of businesses with similar users,
    based on the Jaccard similarity metric.

    Algorithm Steps:
    1. Unpack the prepared data containing the business-to-user mapping and user index mapping.
    2. Generate a set of hash functions.
    3. Compute the Minhash Signature for each business.
    4. Divide the signature matrix into bands.
    5. Group businesses into bands based on their Minhash Signature.
    6. Find candidate pairs of businesses within each band.
    7. Calculate the Jaccard similarity for candidate pairs.
    8. Filter pairs with similarity above a threshold (e.g., 0.5).
    9. Sort the results by business ID pairs.
    10. Return the RDD containing the Jaccard similarity results for candidate business pairs.

    Args:
        prepared_data (tuple): A tuple containing the business-to-user mapping RDD and user index mapping dictionary.

    Returns:
        RDD: An RDD containing the Jaccard similarity results for candidate business pairs.
    """
    # Unpack prepared data
    business_to_user, usr_to_idx = prepared_data

    # Generate Hash functions
    NUM_BINS = len(usr_to_idx)
    hash_func_params = generate_hash_function_params(NUM_BINS, NUM_HASH_FUNCTIONS)

    # Compute Minhash Signature
    minhash_sign = business_to_user.mapValues(lambda users: [usr_to_idx[user] for user in users]).mapValues(
        lambda users: build_minhash_signature_matrix(hash_func_params, users, NUM_BINS)
    )

    # Divide signature matrix into bands
    bands = (
        minhash_sign.flatMap(
            lambda x: [((i, tuple(x[1][i * ROWS_PER_BAND : (i + 1) * ROWS_PER_BAND])), x[0]) for i in range(BANDS)]
        )
        .groupByKey()
        .mapValues(list)
        .filter(lambda x: len(x[1]) > 1)
    )

    # Find the business candidate pairs
    candidates = bands.map(lambda x: sorted(x[1])).flatMap(lambda x: list(combinations(x, 2))).distinct()

    # Calculate Jaccard Similirality for pairs
    bus_to_user_dict = business_to_user.collectAsMap()

    jaccard_sim_results = (
        candidates.map(lambda x: jaccard_similarity(x, bus_to_user_dict))
        .filter(lambda x: x[1] >= 0.5)
        .sortByKey()
        .map(lambda x: [x[0][0], x[0][1], x[1]])
    )
    return jaccard_sim_results


def task1(input_file_name, output_file_name):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 1")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        # Read the input data
        data = spark.textFile(input_file_name)
        prepared_data = prepare_dataset(data)

        # Compute Jaccard similarity using LSH
        jaccard_sim_results = jaccard_based_lsh(prepared_data)

        # Write header and results to a CSV file
        header = ["business_id_1", "business_id_2", "similarity"]
        with open(output_file_name, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(jaccard_sim_results.collect())

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit task1.py <input_file_name> <output_file_name>")
        sys.exit(1)

    # Read input parameters
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    task1(input_file_path, output_file_path)

# task1("HW3StudentData/yelp_train.csv", "t1.csv")
