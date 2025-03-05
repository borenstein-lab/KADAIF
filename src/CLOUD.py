import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.spatial.distance import cdist
import copy


def CLOUD(ra_data, test_data=None, k=None, distance_method="braycurtis"):
    if distance_method == "braycurtis":
        bray_curtis_distances = pdist(ra_data, metric='braycurtis')
        bray_curtis_matrix = squareform(bray_curtis_distances)
        distance_df = pd.DataFrame(
            bray_curtis_matrix,
            index=ra_data.index,
            columns=ra_data.index)
        if test_data is not None:
            test_dist_to_train = pd.DataFrame(cdist(ra_data, test_data, metric='braycurtis'))  # columns are test
            test_dist_to_train.index = ra_data.index
            test_dist_to_train.columns = test_data.index

    if k is None:
        k = int(distance_df.shape[0] * 0.05)

    mean_dist = np.mean(np.array(distance_df)[np.tril_indices_from(distance_df, k=-1)])

    nearest_neighbors = {}
    p_values = {}

    for sample in distance_df.index:
        # Sort distances for the sample and get indices of the k smallest (excluding itself)
        sorted_neighbors = distance_df.loc[sample].sort_values()
        nearest_neighbors[sample] = sorted_neighbors.iloc[1:k + 1].mean() / mean_dist
    if test_data is None:
        return nearest_neighbors, p_values

    test_nn_vals = list(nearest_neighbors.values())
    nearest_neighbors_test_dist = {}
    for sample in test_dist_to_train.columns:
        sorted_neighbors = test_dist_to_train[sample].sort_values()
        nearest_neighbors_test_dist[sample] = sorted_neighbors.iloc[:k].mean() / mean_dist
        p_values[sample] = np.sum([nearest_neighbors_test_dist[sample] >= val for val in test_nn_vals]) / len(
            test_nn_vals)

    return nearest_neighbors_test_dist, p_values


