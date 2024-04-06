# %%writefile task2.py
import sys
import time
from collections import defaultdict

from pyspark import SparkConf, SparkContext


def read_csv(spark: SparkContext, file_path: str):
    """
    Read CSV file into DataFrame
    """
    data = spark.textFile(file_path)
    header = data.first()
    data = data.filter(lambda row: row != header).map(lambda row: row.split(","))
    return data


def construct_graph(data, filter_threshold: int):
    """
    Function to construct the social network graph.
    """
    # Create user to business map
    user2bus = data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set)

    # Create edges RDD
    edges = (
        # Permutation for user pairs eg. [(user1, user2), (user2, user1), ...]
        user2bus.cartesian(user2bus)
        # Filter out pairs with same users
        .filter(lambda pair: pair[0][0] != pair[1][0])
        # Find common business by business set intersection
        .map(lambda pair: ((pair[0][0], pair[1][0]), len(pair[0][1] & pair[1][1])))
        # Filter by threshold
        .filter(lambda data: data[1] >= filter_threshold)
        # Extract user pairs
        .map(lambda data: (data[0][0], data[0][1]))
        # Remove duplicate edges
        .distinct()
    )

    # Create nodes or vertices RDD
    # vertices = edges.map(lambda x: x[0]).cache()

    # Create adjacency matrix RDD
    adj_mat = (
        edges
        # .flatMap(lambda edge: [(edge[0], edge[1]), (edge[1], edge[0])])  # Add both directions of the edge
        .groupByKey()  # Group by user
        .mapValues(set)  # Convert the iterable of neighbors to a set
        .cache()
    )

    return adj_mat.collectAsMap()


def calculate_betweenness(adj_mat):
    """
    Calculate betweenness for each edge in the graph.
    """

    def bfs(graph, source):
        """
        Breadth-first search to calculate shortest paths and number of shortest paths from a source node.
        """
        parent = defaultdict(set)

        level = {}
        level[source] = 0

        num_shortest_paths = defaultdict(float)
        num_shortest_paths[source] = 1

        path = []

        queue = [source]

        visited = set()
        visited.add(source)

        while queue:
            current_node = queue.pop(0)
            path.append(current_node)
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor].add(current_node)
                    num_shortest_paths[neighbor] += num_shortest_paths[current_node]
                    level[neighbor] = level[current_node] + 1
                elif level[neighbor] == level[current_node] + 1:
                    parent[neighbor].add(current_node)
                    num_shortest_paths[neighbor] += num_shortest_paths[current_node]

        return parent, level, num_shortest_paths, path

    def accumulate_edge_weights(path, parent, num_shortest_paths):
        node_weights = {node: 1 for node in reversed(path)}
        edge_weights = defaultdict(float)

        for node in reversed(path):
            for parent_node in parent[node]:
                temp_weight = node_weights[node] * (num_shortest_paths[parent_node] / num_shortest_paths[node])
                node_weights[parent_node] += temp_weight
                edge = tuple(sorted([node, parent_node]))
                edge_weights[edge] += temp_weight / 2

        return edge_weights

    betweenness = defaultdict(float)
    for node in adj_mat:
        parent, level, num_shortest_paths, path = bfs(adj_mat, node)
        edge_weights = accumulate_edge_weights(path, parent, num_shortest_paths)
        for edge, weight in edge_weights.items():
            betweenness[edge] += weight

    betweenness = sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))
    return betweenness


def save_betweeness_data(data: list, path: str):
    with open(path, "w") as f:
        lines = [f"{str(user)},{round(value, 5)}\n" for user, value in data]
        f.writelines(lines)


def task2(
    filter_threshold: int, input_file_path: str, betweenness_output_file_path: str, community_output_file_path: str
):
    # Initialize Spark
    conf = SparkConf().setAppName("Task 2")
    spark = SparkContext(conf=conf).getOrCreate()
    spark.setLogLevel("ERROR")

    try:
        start_time = time.time()

        data = read_csv(spark, input_file_path)

        # Construct graph and get the adjacency matrix
        adj_mat = construct_graph(data, filter_threshold)

        # Calculate betweenness centrality
        betweenness = calculate_betweenness(adj_mat)
        save_betweeness_data(betweenness, betweenness_output_file_path)

        execution_time = time.time() - start_time
        print(f"Duration: {execution_time}\n")

    finally:
        # Stop Spark
        spark.stop()


if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 5:
        print(
            "Usage: spark-submit task2.py <input_file_path> <betweenness_output_file_path> <community_output_file_path>"
        )
        sys.exit(1)

    # Parse command-line arguments
    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    # Call task2 function
    task2(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path)

# task2(5, Path.input_csv_file, Path.task2_bw_output, Path.task2_cm_output)
