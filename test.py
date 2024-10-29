def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
X = [1,2,3,4]
labels = [2,3,1,2]
print(X[labels == 10])