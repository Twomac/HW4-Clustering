import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """


    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # all folloiwing code suggested by VScode
        # calculate distance matrix between all observations in X
        dist_matrix = cdist(X, X)  # n x n matrix of dists b/w obs

        # calculate silhouette score for each observation
        silhouette_scores = np.zeros(X.shape[0])  # initialize array to store silhouette scores for each observation

        for observation in range(X.shape[0]):
            same_cluster = (y == y[observation])  # boolean array indicating which observations are in the same cluster as observation i
            same_cluster[observation] = False  # exclude the observation itself from the same cluster calculation, I was doing this earlier and it broke m sk test!

            a_i = np.mean(dist_matrix[observation, same_cluster])  # average distance from observation i to all other observations in the same cluster
            b_i = np.min([np.mean(dist_matrix[observation, y == label]) for label in np.unique(y) if label != y[observation]])  # minimum average distance from observation i to all observations in other clusters

            silhouette_scores[observation] = (b_i - a_i) / max(a_i, b_i)  # silhouette score for observation i

        return silhouette_scores
