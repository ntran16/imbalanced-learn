"""Class to perform under-sampling by generating centroids based on
clustering."""
from __future__ import print_function
from __future__ import division

import hdbscan

import numpy as np

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

from ..base import BaseMulticlassSampler


class ClusterCentroids(BaseMulticlassSampler):
    """Perform under-sampling by generating centroids based on
    clustering methods.

    Experimental method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans algorithm.
    This algorithm keeps N majority samples by fitting the KMeans algorithm
    with N cluster to the majority class and using the coordinates of the N
    cluster centroids as the new majority samples.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    **kwargs : keywords
        Parameter to use for the KMeans object.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    This class support multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import ClusterCentroids
    >>> X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
    ...                            n_informative=3, n_redundant=1, flip_y=0,
    ...                            n_features=20, n_clusters_per_class=1,
    ...                            n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> cc = ClusterCentroids(random_state=42)
    >>> X_res, y_res = cc.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 100, 1: 100})

    """

    def __init__(self, ratio='auto', random_state=None, n_jobs=-1, **kwargs):
        super(ClusterCentroids, self).__init__(ratio=ratio)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """
        random_state = check_random_state(self.random_state)

        # Compute the number of cluster needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # Create the clustering object
        kmeans = KMeans(n_clusters=num_samples, random_state=random_state)
        kmeans.set_params(**self.kwargs)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it.
            if key == self.min_c_:
                continue

            # Find the centroids via k-means
            kmeans.fit(X[y == key])
            centroids = kmeans.cluster_centers_

            # Concatenate to the minority class
            X_resampled = np.concatenate((X_resampled, centroids), axis=0)
            y_resampled = np.concatenate((y_resampled, np.array([key] *
                                                                num_samples)),
                                         axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(
            y_resampled))

        return X_resampled, y_resampled


class DensityBasedClustering(BaseMulticlassSampler):
    """Perform under-sampling by removing sample based on HDBSCAN algorithm.

    HDBSCAN algorithm can be used to cluster classes and remove samples which
    are the least probable to belong to any cluster and that can be consider
    as noisy or corrupted samples.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    This class support multi-class.

    """

    def __init__(self, ratio='auto', random_state=None, return_indices=False,
                 alpha=1.0, leaf_size=40, metric='euclidean',
                 min_cluster_size=5, min_samples=None, p=None, n_jobs=-1):
        super(DensityBasedClustering, self).__init__(ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.alpha = alpha
        self.leaf_size = leaf_size
        self.metric = metric
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.p = p
        self.n_jobs = n_jobs

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        # Set the HDBSCAN object
        clusterer = hdbscan.HDBSCAN(alpha=self.alpha, leaf_size=self.leaf_size,
                                    metric=self.metric,
                                    min_cluster_size=self.min_cluster_size,
                                    min_samples=self.min_samples, p=self.p,
                                    core_dist_n_jobs=self.n_jobs)

        # Compute the number of cluster needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.flatnonzero(y == self.min_c_)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it.
            if key == self.min_c_:
                continue

            # Fit the data to find the possible outliers for this class
            clusterer.fit(X[y == key])

            # Find the probabilities threshold to find such that the number of
            # samples enforce the imbalance ratio given by the user
            threshold = np.percentile(
                clusterer.probabilities_,
                (1. - (num_samples / self.stats_c_[self.maj_c_])) * 100.)

            # Select the point with higher probabily
            idx_curr_class = np.flatnonzero(y == key)
            sel_idx = idx_curr_class[np.flatnonzero(
                clusterer.probabilities_ >= threshold)]

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_under = np.concatenate((idx_under, sel_idx), axis=0)

            # Concatenate to the minority class
            X_resampled = np.concatenate((X_resampled, X[sel_idx]), axis=0)
            y_resampled = np.concatenate((y_resampled, np.array([key] *
                                                                sel_idx.size)),
                                         axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(
            y_resampled))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
