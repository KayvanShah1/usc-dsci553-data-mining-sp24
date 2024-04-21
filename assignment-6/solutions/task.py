import sys
import time
import traceback

import numpy as np
from sklearn.cluster import KMeans

LARGE_K_FACTOR = 5


def read_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            row = line.strip().split(",")
            data.append([float(value) for value in row])
    return np.array(data)


def get_random_splits(arr: np.array):
    """
    Splits the input array into 5 roughly equal parts after shuffling.
    """
    np.random.shuffle(arr)
    return np.array_split(arr, 5)


def cluster_points(data: np.array, n_cluster: int, factor: int):
    """
    Clusters the data points using KMeans algorithm.

    Args:
        data (np.array): The input data array containing points.
        n_cluster (int): The desired number of clusters.

    Returns:
        Dict[int, List[int]]: A dictionary where keys represent cluster IDs and values contain the indices of points
        assigned to each cluster.
    """
    model = KMeans(n_clusters=factor * n_cluster)
    model.fit(data[:, 2:])

    clusters = {}
    for i, cluster_id in enumerate(model.labels_):
        if cluster_id not in clusters.keys():
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    return clusters


def process_cluster_data(clusters, data, track_set, check_len=False):
    """
    Process cluster data and update the track_set dictionary with summary statistics.

    Args:
    - clusters (dict): A dictionary containing cluster IDs as keys and indices of data points belonging to each cluster
    as values.
    - data (numpy.ndarray): An array containing the data points.
    - track_set (dict): A dictionary containing track set information, including summary statistics, centroids,
    variance, and data points.
    - check_len (bool, optional): A flag indicating whether to check if the number of data points in a cluster is less
    than or equal to 1. Defaults to False.

    Returns:
    - dict: The updated track_set dictionary containing summary statistics for each cluster.

    """
    for cluster_id, idx in clusters.items():
        n = len(idx)
        if check_len and n <= 1:
            continue

        feats = data[idx, 2:]
        row_ids = data[idx, 0]

        SUM = np.sum(feats, axis=0)
        SUMSQ = np.sum(np.square(feats), axis=0)

        ctr = SUM / n
        pts = row_ids.astype(int).tolist()
        var = np.sqrt(np.subtract(SUMSQ / n, np.square(ctr)))

        track_set["smr"][cluster_id] = [n, SUM, SUMSQ]
        track_set["pts"][cluster_id] = pts
        track_set["ctr"][cluster_id] = ctr
        track_set["var"][cluster_id] = var

    return track_set


def m_dist_pt(feat, ctr, var):
    """
    Calculate Mahalanobis distance between a feature vector and cluster centroid.

    Mahalanobis Distance (MD) Formula:
        MD = sqrt(sum(((feat - ctr) / var) ** 2))

    Args:
    - feat (numpy.ndarray): Feature vector.
    - ctr (numpy.ndarray): Cluster centroid.
    - var (numpy.ndarray): Variance vector.

    Returns:
    - float: Mahalanobis distance.
    """
    diff = feat - ctr
    z = np.square(diff / var)
    return np.sqrt(np.sum(z))


def m_dist_clusters(ctr1, ctr2, var):
    """
    Calculate Mahalanobis distance between centroids of two clusters.

    Mahalanobis Distance (MD) Formula:
        MD = sqrt(sum(((ctr1 - ctr2) / var) ** 2))

    Args:
    - ctr1 (numpy.ndarray): Centroid of the first cluster.
    - ctr2 (numpy.ndarray): Centroid of the second cluster.
    - var (numpy.ndarray): Variance vector.

    Returns:
    - float: Mahalanobis distance.
    """
    diff = ctr1 - ctr2
    z = np.square(diff / var)
    return np.sqrt(np.sum(z))


def reassign_cluster(feat, t_set):
    """
    Reassign a feature vector to the nearest cluster based on Mahalanobis distance.

    Args:
    - feat (numpy.ndarray): Feature vector to be reassigned.
    - t_set (dict): Track set containing cluster information.

    Returns:
    - tuple: Mahalanobis distance and ID of the nearest cluster.
    """
    max_dist = np.inf
    cluster = -1
    for cluster_id, _ in t_set["smr"].items():
        ctr = t_set["ctr"][cluster_id]
        var = t_set["var"][cluster_id]

        m_dist = m_dist_pt(feat, ctr, var)
        if m_dist < max_dist:
            max_dist = m_dist
            cluster = cluster_id

    return max_dist, cluster


def update_cluster_summary(feat, t_set, cluster, row_id):
    """
    Update the summary statistics of a cluster after reassigning a feature vector.

    Args:
    - feat (numpy.ndarray): The feature vector being reassigned to the cluster.
    - t_set (dict): The track set containing cluster information.
    - cluster (int): The ID of the cluster being updated.
    - row_id: The ID of the data row corresponding to the reassigned feature vector.

    Returns:
    - dict: The updated track set with the cluster summary statistics updated.
    """
    n = t_set["smr"][cluster][0] + 1
    SUM = np.add(t_set["smr"][cluster][1], feat)
    SUMSQ = np.add(t_set["smr"][cluster][2], np.square(feat))

    ctr = SUM / n
    var = np.sqrt(np.subtract(SUMSQ / n, np.square(ctr)))

    t_set["smr"][cluster] = [n, SUM, SUMSQ]
    t_set["ctr"][cluster] = ctr
    t_set["var"][cluster] = var
    t_set["pts"][cluster].append(row_id)

    return t_set


def merge_clusters(t_set1, t_set2, threshold_dist, inverse=False):
    """
    Merge clusters between two track sets based on Mahalanobis distance.

    Args:
    - t_set1 (dict): The first track set containing cluster information.
    - t_set2 (dict): The second track set containing cluster information.
    - threshold_dist (float): The threshold mahalanobis distance for merging clusters.
    - inverse (bool, optional): A flag indicating whether to merge clusters in reverse order. Defaults to False.

    Returns:
    - tuple: A tuple containing the updated track sets after merging clusters.
    """
    new_assignment = {}
    for c1 in t_set1["smr"].keys():
        cluster = -1
        for c2 in t_set2["smr"].keys():
            if c1 != c2:
                # Calculate Mahalanobis distance between centroids of clusters
                m_dist1 = m_dist_clusters(t_set1["ctr"][c1], t_set2["ctr"][c2], t_set2["var"][c2])
                m_dist2 = m_dist_clusters(t_set2["ctr"][c2], t_set1["ctr"][c1], t_set1["var"][c1])
                m_dist = min(m_dist1, m_dist2)
                # Check if distance is below threshold for merging
                if m_dist < threshold_dist:
                    threshold_dist = m_dist
                    cluster = c2
        new_assignment[c1] = cluster

    # Merge clusters according to new assignment
    for c1, c2 in new_assignment.items():
        if c1 in t_set1["smr"] and c2 in t_set2["smr"]:
            if c1 != c2:
                # Update cluster summary statistics
                n = t_set1["smr"][c1][0] + t_set2["smr"][c2][0]
                SUM = np.add(t_set1["smr"][c1][1], t_set2["smr"][c2][1])
                SUMSQ = np.add(t_set1["smr"][c1][2], t_set2["smr"][c2][2])

                ctr = SUM / n
                var = np.sqrt(np.subtract(SUMSQ / n, np.square(ctr)))

                # Update track set with merged cluster information
                t_set2["smr"][c2] = [n, SUM, SUMSQ]
                t_set2["ctr"][c2] = ctr
                t_set2["var"][c2] = var
                t_set2["pts"][c2].extend(t_set1["pts"][c1])

                # Determine which cluster to remove
                cluster_to_pop = c1 if inverse else c2

                # Remove the cluster from the appropriate track set
                t_set1["smr"].pop(cluster_to_pop)
                t_set1["ctr"].pop(cluster_to_pop)
                t_set1["var"].pop(cluster_to_pop)
                t_set1["pts"].pop(cluster_to_pop)

    return t_set1, t_set2


def update_round_results(round_res: list, round_num: int, ds_smr: dict, cs_smr: dict, r_set: set):
    """
    Write intermediate clustering results to a file.

    Parameters:
        output_path (str): The path to the output file.
        round_num (int): The current round number of clustering.
        ds_smr (dict): A dictionary containing parameters for the Discard Set clusters.
        cs_params_dict (dict): A dictionary containing parameters for the Compression Set clusters.
        r_set (set): A set containing isolated points in the Retained Set.
    """
    num_ds = sum(value[0] for value in ds_smr.values())
    num_cs = sum(value[0] for value in cs_smr.values())

    result_str = f"Round {round_num}: {num_ds},{len(cs_smr)},{num_cs},{len(r_set)}\n"
    round_res.append(result_str)

    return round_res


def save_results(round_results, final_results, output_path):
    with open(output_path, "w") as f:
        f.write("The intermediate results:\n")
        f.writelines(round_results)

        f.write("\n")
        f.write("The clustering results:\n")
        for k, v in final_results.items():
            f.write(f"{int(k)},{int(v)}\n")


def BFR(data: np.array, n_cluster: int):
    # Create Dictionaries to keep track of data
    # "smr": Summary statistics, "ctr": Centroids, "var": Variances, "pts": Data points indexes.
    d_set = {"smr": {}, "ctr": {}, "var": {}, "pts": {}}  # Discarded Set
    c_set = {"smr": {}, "ctr": {}, "var": {}, "pts": {}}  # Compressed Set

    # Store results
    round_res = []
    final_res = {}
    # flag = False

    # Step 1: Load 20% of the data randomly
    data = get_random_splits(data)
    split1 = data[0]

    # Step 2: Run K-Means (e.g., from sklearn) with a large K (e.g., 5 times of the number of the input clusters)
    # on the data in memory using the Euclidean distance as the similarity measurement
    clusters = cluster_points(split1, n_cluster, LARGE_K_FACTOR)

    # Step 3: In the K-Means result from Step 2, move all the clusters that contain only one point to RS (outliers).
    r_set = {idx[0] for idx in clusters.values() if len(idx) == 1}
    ds_data = np.delete(split1, list(r_set), axis=0)

    # Step 4: Run K-Means again to cluster the rest of the data points with K = the number of input clusters.
    clusters = cluster_points(ds_data, n_cluster, 1)

    # Step 5: Use the K-Means result from Step 4 to generate the DS clusters (i.e., discard their points and
    # generate statistics).
    d_set = process_cluster_data(clusters, ds_data, d_set)

    # The initialization of DS has finished, so far, you have K numbers of DS clusters (from Step 5) and some
    # numbers of RS (from Step 3).
    rs_data = split1[list(r_set), :]
    if len(r_set) >= LARGE_K_FACTOR * n_cluster:
        clusters = cluster_points(rs_data, n_cluster, LARGE_K_FACTOR)
        r_set = {idx[0] for idx in clusters.values() if len(idx) == 1}
        c_set = process_cluster_data(clusters, rs_data, c_set, check_len=True)

    # Store intermediate results for 1st round
    round_res = update_round_results(round_res, 1, d_set["smr"], c_set["smr"], r_set)

    THRESHOLD_DIST = 2 * np.sqrt(split1.shape[1] - 2)

    # Repeat Step 7 - 12
    for round in range(2, 6):
        # Step 7: Load another 20% of the data randomly.
        split = data[round - 1]

        for idx, value in enumerate(split):
            row_id = int(value[0])
            feat = value[2:]

            # Step 8: For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them
            # to the nearest DS clusters if the distance is < 2 * sqrt(𝑑).
            max_dist, cluster = reassign_cluster(feat, d_set)
            if max_dist < THRESHOLD_DIST and cluster != -1:
                d_set = update_cluster_summary(feat, d_set, cluster, row_id)

            # Step 9: For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign
            # the points to the nearest CS clusters if the distance is < 2 * sqrt(𝑑)
            else:
                max_dist, cluster = reassign_cluster(feat, c_set)
                if max_dist < THRESHOLD_DIST and cluster != -1:
                    c_set = update_cluster_summary(feat, c_set, cluster, row_id)

                # Step 10: For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS.
                else:
                    r_set.add(idx)

        # Step 11: Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to generate
        # CS (clusters with more than one points) and RS (clusters with only one point).
        rs_split = split[list(r_set), :]
        if len(r_set) >= LARGE_K_FACTOR * n_cluster:
            clusters = cluster_points(rs_split, n_cluster, LARGE_K_FACTOR)
            r_set = {idx[0] for idx in clusters.values() if len(idx) == 1}
            c_set = process_cluster_data(clusters, rs_split, c_set, check_len=True)

        # Step 12: Merge CS clusters that have a Mahalanobis Distance < 2 * sqrt(𝑑).
        # if flag:
        c_set, _ = merge_clusters(c_set, c_set, THRESHOLD_DIST, inverse=False)

        if round == 5:
            c_set, d_set = merge_clusters(c_set, d_set, THRESHOLD_DIST, inverse=True)

        # Store intermediate results for every round
        round_res = update_round_results(round_res, round, d_set["smr"], c_set["smr"], r_set)

    if len(r_set) > 0:
        rs_data = data[4][list(r_set), 0]
        r_set = set([int(n) for n in rs_data])

    # Process points in discarded set
    for cluster_id in d_set["smr"].keys():
        for pt in d_set["pts"][cluster_id]:
            final_res[pt] = cluster_id

    # Process points in compressed set
    for cluster_id in c_set["smr"].keys():
        for pt in c_set["pts"][cluster_id]:
            final_res[pt] = -1

    # Process points in retained set
    for pt in r_set:
        final_res[pt] = -1

    return round_res, final_res


def task(input_path: str, n_cluster: int, output_path: str):
    # np.random.seed(37)
    start_time = time.time()
    try:
        data = read_data(input_path)

        round_results, final_results = BFR(data, n_cluster)

        save_results(round_results, final_results, output_path)
    except Exception:
        traceback.print_exc()

    execution_time = time.time() - start_time
    print(f"Duration: {execution_time}\n")


if __name__ == "__main__":
    # Check if correct number of command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python task.py <input_filename> <n_cluster> <output_filename>")
        sys.exit(1)

    # Parse command-line arguments
    input_path = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_path = sys.argv[3]

    # Call task1 function
    task(input_path, n_cluster, output_path)

# task(Path.input_csv_file, 5, Path.task_output_file)
