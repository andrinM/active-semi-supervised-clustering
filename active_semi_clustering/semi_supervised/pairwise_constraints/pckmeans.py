""" This is an implementation of a PCKMeans algorithm
"""

import numpy as np

from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints


class PCKMeans:
    def __init__(self, n_clusters=3, max_iter=100, w=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w
    """ The first iteration takes the initialized clusters. Then each data point gets assigned
    to a cluster. We then safe those cluster_centers in prev_cluster_centers. Then for each cluster
    We calculate a new cluster_center, based on the freshly asigned data points (mean of those).
    This is done with the function _get_cluster_centers. We then compare the new centers with the
    old ones.

    Keyword arguments:
    converged --This returns True if all elements in the difference array are within the defined tolerances
    (atol=1e-6 and rtol=0) of the corresponding elements in the zero array.
    If any element is outside this tolerance, it returns False.
    """
    def fit(self, X, y=None, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize centroids
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

        # Repeat until convergence
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Check for convergence, this compares the difference with the zero matrix
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self
    
    """ This function returns a set of cluster_centers equal to the size of k.
        neighboorhood_centers is the mean of each center based on the feature values
        neighboorhood_size is the amount of data points per neighboorhood
        If #neighboorhods > k then select the k biggest neighborhoods
        If #neighboorhods > 0 then cluster_centers = neighborhood_centers
        If #neighboorhods < k then add k-#neighboorhods random centroids
    """
    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])
#neighborhood_centers is the mean of all data points from this neighborhood.
        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers
    
    """Keyword arguments:
    X -- data set
    x_i -- current inspected data point
    centroids -- list of cluster_centers
    labels -- list #rows filled with -1
    c_i -- index of current cluster (calculate cost if x_i is in c_i)

    The ml_penalty gets calculated as follows: We look at all data points must-link with
    x_i (called y_i). Then for each y_i figure out if it allready has a
    cluster (if labels[y_i] != -1) and if it is the same cluster c_i that we currently
    look at. If y_i has allready a cluster and its not c_i, then we get a penalty.

    The cl_penalty gets added, if a data point who y_i cannot-link with x_i is already in cluster
    c_i
    """
    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w):
        distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_penalty = 0
        for y_i in ml_graph[x_i]: # iterate over all must-links from data point x_i
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w

        return distance + ml_penalty + cl_penalty
    

    """ Keyword arguments:
    labels -- initially a list with #data points filled with -1, this list will have the assigned clusters
    index -- to randomlly choose a sequence of data point indices, to make the decision of when
    which data point gets clustered random.
    x_i -- current data point we want to asign a cluster
    c_i -- current cluster we look at.

    We itterate over the number of clusters (k) in the function. We calculate with the objective
    function the cost of x_i to each centroid. Then we take the minimum with argmin
 
    """
    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1) # numpy array with #rows filled with -1

        index = list(range(X.shape[0])) # list with 0,1 ...,n where n is the #rows
        np.random.shuffle(index)
        for x_i in index: # for every data point with every cluster
            labels[x_i] = np.argmin([self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels

    """ Getting all elements from X that are in cluster i and calculate the mean. The mean is
    the centroid for the next iteration
    """
    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
