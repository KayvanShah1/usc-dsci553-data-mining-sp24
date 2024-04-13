import csv
import random
import sys
import time

from blackbox import BlackBox


def reservoir_sampling(input_path: str, blackbox: BlackBox, num_of_asks: int, stream_size: int):
    results = []

    users_found = []
    seq_num = 0

    for i in range(num_of_asks):
        stream_usrs = blackbox.ask(input_path, stream_size)

        for usr in stream_usrs:
            seq_num += 1

            # For the first 100 users, add them directly to the reservoir
            if len(users_found) < 100:
                users_found.append(usr)
            elif random.random() < 100 / seq_num:
                replace_index = random.randint(0, 99)
                users_found[replace_index] = usr

            # Output the current stage of the reservoir after every 100 users
            if seq_num % 100 == 0:
                results.append([seq_num] + users_found[::20])

    return results


def save_output(output_file_name, results):
    header = ["seqnum", "0_id", "20_id", "40_id", "60_id", "80_id"]

    with open(output_file_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)


def task3(input_path: str, stream_size: int, num_of_asks: int, output_path: str):
    try:
        start_time = time.time()

        # Initialize BlackBox
        blackbox = BlackBox()

        # Apply algorithm on stream of users
        results = reservoir_sampling(input_path, blackbox, num_of_asks, stream_size)

        # Write results to output file
        save_output(output_path, results)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Check if correct number of command-line arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python task3.py <input_filename> <stream_size> <num_of_asks> <output_filename>")
        sys.exit(1)

    random.seed(553)

    # Parse command-line arguments
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    # Call task1 function
    task3(input_path, stream_size, num_of_asks, output_path)

# task3(Path.input_csv_file, 100, 30, Path.task3_output)
