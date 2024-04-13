import binascii
import csv
import random
import sys
import time

from blackbox import BlackBox

FILTER_ARRAY_LENGTH = 997
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


def calculate_ground_truth(stream_users, ground_truth):
    for user in stream_users:
        ground_truth.add(user)

    return len(ground_truth)


def calculate_estimation(usr_hash):
    sum_estimate = 0

    for h in range(NUM_HASHES):
        temp = [int(value[h]) for value in usr_hash.values()]

        max_t_zero = 0
        for value in temp:
            tmp_str = bin(value)[2:]
            wo_zero = tmp_str.rstrip("0")
            if max_t_zero < len(tmp_str) - len(wo_zero):
                max_t_zero = len(tmp_str) - len(wo_zero)

        sum_estimate += 2**max_t_zero
    return sum_estimate // NUM_HASHES


def flajolet_martin(input_path: str, blackbox: BlackBox, num_of_asks: int, stream_size: int):
    results = []

    # Fetch stream and perform Bloom Filtering for each batch
    for i in range(num_of_asks):
        ground_truth = set()
        usr_hash = dict()

        stream_usrs = blackbox.ask(input_path, stream_size)
        len_ground_truth = calculate_ground_truth(stream_usrs, ground_truth)

        for user in stream_usrs:
            usr_hash[user] = myhashs(user)

        estimate = calculate_estimation(usr_hash)
        results.append([i, len_ground_truth, estimate])

    return results


def save_output(output_file_name, results):
    header = ["Time", "Ground Truth", "Estimation"]
    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


def task2(input_path: str, stream_size: int, num_of_asks: int, output_path: str):
    try:
        start_time = time.time()

        # Initialize BlackBox
        blackbox = BlackBox()

        # Apply bloom filter on stream of users
        results = flajolet_martin(input_path, blackbox, num_of_asks, stream_size)

        # Write results to output file
        save_output(output_path, results)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Check if correct number of command-line arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python task2.py <input_filename> <stream_size> <num_of_asks> <output_filename>")
        sys.exit(1)

    # Parse command-line arguments
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    # Call task1 function
    task2(input_path, stream_size, num_of_asks, output_path)

# task2(Path.input_csv_file, 300, 30, Path.task2_output)
