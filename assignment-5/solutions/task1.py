import binascii
import csv
import random
import sys
import time

from blackbox import BlackBox
from pyspark import SparkConf, SparkContext

FILTER_ARRAY_LENGTH = 69997
ENCODING = "utf-8"
NUM_HASHES = 50
PRIME_NUMBER = 1e9 + 7


def generate_hash_function_params(max_range, count):
    """Generate random hash function parameters within a specified range."""
    a = random.sample(range(1, max_range), count)  # Random coefficient 'a'
    b = random.sample(range(1, max_range), count)  # Random intercept 'b'
    return list(zip(a, b))


def hash_user(user, params):
    """Hash an item using given hash function parameters.
    Calculate hash value using the formula: ((a * item + b) % PRIME_NUMBER) % num_bins
    """
    user = int(binascii.hexlify(user.encode("utf8")), 16)
    hash_val = ((params[0] * user + params[1]) % PRIME_NUMBER) % FILTER_ARRAY_LENGTH
    return hash_val


def myhashs(user):
    hash_funcs = generate_hash_function_params(FILTER_ARRAY_LENGTH, NUM_HASHES)
    return [hash_user(user, hash_funcs[i]) for i in range(NUM_HASHES)]


def bloom_filter(input_path: str, blackbox: BlackBox, num_of_asks: int, stream_size: int):
    results = []
    exist_user = set()
    exist_hash = []

    # Fetch stream and perform Bloom Filtering for each batch
    for i in range(num_of_asks):
        stream_users = blackbox.ask(input_path, stream_size)
        false_positives = 0
        for user in stream_users:
            usr_hashes = myhashs(user)

            if usr_hashes in exist_hash and user not in exist_user:
                false_positives += 1

            exist_hash.append(usr_hashes)
            exist_user.add(user)

        results.append([i, false_positives / stream_size])

    return results


def save_output(output_file_name, results):
    header = ["Time", "FPR"]
    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


def task1(input_path: str, stream_size: int, num_of_asks: int, output_path: str):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 1: Bloom Filter").setMaster("local[*]")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        # Initialize BlackBox
        blackbox = BlackBox()

        # Apply bloom filter on stream of users
        results = bloom_filter(input_path, blackbox, num_of_asks, stream_size)

        # Write results to output file
        save_output(output_path, results)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    # Check if correct number of command-line arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python task1.py <input_filename> <stream_size> <num_of_asks> <output_filename>")
        sys.exit(1)

    # Parse command-line arguments
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    # Call task1 function
    task1(input_path, stream_size, num_of_asks, output_path)

# task1(Path.input_csv_file, 100, 30, Path.task1_output)
