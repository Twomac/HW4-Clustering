import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.labels = None  # cluster label for each observation, corresponding to row index of observations in the input matrix
        self.centroids = None
        self.error = None
        self.learned_mat = None  # store the original matrix that the model was fit on


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        n, m = mat.shape
        assert self.k <= n, "Number of clusters cannot be greater than number of observations"

        # randomly assign cluster labels since we don't know cluster centers yet
        self.labels = np.random.randint(0, self.k, size=n)
        self.learned_mat = mat

        prev_error = np.inf

        # below code suggested by VScode
        for step in range(self.max_iter):
            # calculate centroids based on current cluster labels, labels == cluster uses boolean indexing to only get 
            # the indices of points in labels that match the cluster number, then mean is taken across rows to get the 
            # means of each feature, defining the centroid for that cluster
            self.centroids = np.array([mat[self.labels == cluster].mean(axis=0) for cluster in range(self.k)])  # k x m matrix of centroids

            # calculate distances from each observation to each centroid 
            distances = cdist(mat, self.centroids)  # n x k matrix of dists b/w obs and cent

            # assign new cluster labels based on closest centroid
            # find index of column with min dist for each row, assign that index as the new cluster label for that observation
            new_labels = np.argmin(distances, axis=1)

            # calculate error as mean squared distance to assigned centroid
            self.error = np.mean(np.min(distances, axis=1))

            # check for convergence criteria
            if np.array_equal(new_labels, self.labels) or abs(self.error - prev_error) < self.tol:
                break

            self.labels = new_labels
            prev_error = self.error



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        # suggested by VScode
        if self.centroids is None:
            raise ValueError("Model has not been fit yet, centroids are not available for prediction")
        
        # check that the matrix has the same number of columns as the training data (suggested by VScode)
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("Input matrix has a different number of features than the training data")

        # calculate distances from each observation in mat to each centroid
        distances = cdist(mat, self.centroids)  # n x k matrix of dists b/w obs and cent

        # return cluster labels for each observation in mat (index of closest centroid)
        return np.argmin(distances, axis=1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.error is None:
            raise ValueError("Model has not been fit yet, error is not available")
        # error is already calculated and stored during fitting, so just return that value
        return self.error


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise ValueError("Model has not been fit yet, centroids are not available")
        return self.centroids

