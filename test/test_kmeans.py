# Write your k-means unit tests here
import numpy as np
import pytest

from cluster import KMeans

# all tests suggested by ChatGPT after I explained the edge cases and provided the kmeans and silhouette scripts
# Sorry, I had a busy week and other pressing deadlines

def test_k_zero_raises_value_error():
    with pytest.raises(ValueError):
        KMeans(k=0)

def test_k_greater_than_n_raises_assertion():
    X = np.random.randn(5, 3)  # n = 5
    model = KMeans(k=10)

    with pytest.raises(AssertionError):
        model.fit(X)

def test_high_k_runs():
    n = 50
    m = 4
    k = 45  # very high relative to n

    X = np.random.randn(n, m)
    model = KMeans(k=k, max_iter=10)

    model.fit(X)

    assert model.centroids.shape == (k, m)
    assert model.labels.shape == (n,)


def test_high_dimensional_data_runs():
    n = 30
    m = 1000  # high dimensionality
    k = 3

    X = np.random.randn(n, m)
    model = KMeans(k=k, max_iter=10)

    model.fit(X)

    assert model.centroids.shape == (k, m)


def test_single_dimension_data_runs():
    n = 40
    X = np.random.randn(n, 1)  # single feature
    model = KMeans(k=2, max_iter=10)

    model.fit(X)

    assert model.centroids.shape == (2, 1)
    assert model.labels.shape == (n,)

def test_predict_dimension_mismatch_raises():
    X_train = np.random.randn(20, 3)
    X_test = np.random.randn(10, 2)

    model = KMeans(k=2)
    model.fit(X_train)

    with pytest.raises(ValueError):
        model.predict(X_test)


