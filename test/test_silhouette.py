# write your silhouette score unit tests here
import numpy as np
import pytest
from sklearn.metrics import silhouette_score as sklearn_silhouette

from cluster import KMeans, Silhouette

# test all suggested by ChatGPT

def test_silhouette_matches_sklearn():
    np.random.seed(0)

    X = np.random.randn(50, 4)
    model = KMeans(k=3, max_iter=20)
    model.fit(X)

    labels = model.labels

    # our implementation
    sil = Silhouette()
    my_scores = sil.score(X, labels)
    my_mean = np.mean(my_scores)

    # sklearn implementation
    skl_score = sklearn_silhouette(X, labels)

    assert np.isclose(my_mean, skl_score, atol=1e-6)