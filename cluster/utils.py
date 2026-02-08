import numpy as np
import matplotlib.pyplot as plt

def make_clusters(
        n: int = 500, 
        m: int = 2, 
        k: int = 3, 
        bounds: tuple = (-10, 10),
        scale: float = 1,
        seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    creates some clustered data

    inputs:
        n: int
            number of observations
        m: int
            number of features
        k: int
            number of clusters
        bounds: tuple
            minimum and maximum bounds for cluster grid
        scale: float
            standard deviation of normal distribution
        seed: int
            random seed

    outputs:
        (np.ndarray, np.ndarray)
            returns a 2D matrix of `n` observations and `m` features that are clustered into `k` groups
            returns a 1D array of `n` size that defines the cluster origin for each observation
    """
    np.random.seed(seed)
    assert k <= n

    labels = np.sort(np.random.randint(0, k, size=n))  # assign cluster number to each observation, sorted to ensure that clusters are contiguous in the output matrix, indices corresponding to observation number
    centers = np.random.uniform(bounds[0], bounds[1], size=(k,m)) # assign cluster centers in columns for each of m features, rows corresponding to cluster number
    mat = np.vstack([
        np.random.normal(
            loc=centers[idx],  # center of each cluster for each of the m features is the mean of each gaussian
            scale=scale,
            size=(np.sum(labels==idx), m)) # for each cluster, generate a normal distribution of observations around the cluster center (for each of the m features), with the number of observations corresponding to the number of times that cluster number appears in the labels array
        for idx in np.arange(0, k)])  # idx is cluster number

    return mat, labels


def plot_clusters(
        mat: np.ndarray, 
        labels: np.ndarray, 
        filename: str =None):
    """
    inputs:
        mat: np.ndarray
            a 2D matrix where each row is an observation and each column is a feature
        labels: np.ndarray
            a 1D array where each value represents an integer cluster that an observation belongs to
        filename: str
            an optional value to save a figure to a file
    """

    plt.figure(figsize=(5,5), dpi=200)
    plt.scatter(
        mat[:,0], 
        mat[:,1], 
        c=labels)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_multipanel(
        mat: np.ndarray,
        truth: np.ndarray,
        pred: np.ndarray,
        score: np.ndarray,
        filename: str = None):
    """
    Plots a multipanel figure visualizing the efficiency of truth, prediction, 
    and silhouette scoring on a provided dataset

    inputs:
        mat: np.ndarray
            a 2D matrix where each row is an observation and each column is a feature
        truth: np.ndarray
            a 1D array where each value represents a true integer cluster that an observation belongs to
        pred: np.ndarray
            a 1D array where each value represents a predicted integer cluster than an observation belongs to
        score: np.ndarray
            a 1D array where each value represents a float for the silhouette score of that observation
        filename: str
            an optional value to save a figure to a file
    """

    fig, axs = plt.subplots(1, 3, figsize=(9,3), dpi=200)
    
    cvars = [truth, pred, score]
    names = ["True Cluster Labels", "Predicted Cluster Labels", "Silhouette Scores"]
    cmaps = [None, None, "seismic"]
    for idx, ax in enumerate(axs):
        sub = ax.scatter(
            mat[:,0],
            mat[:,1],
            c=cvars[idx],
            cmap=cmaps[idx])
        ax.set_title(names[idx])
        if idx == 2:
            plt.colorbar(sub, ax=ax)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

