import numpy as np
neighborhoods = [[0],[1,2,3], [4,5,6], [7,8]]
X = [[1,2], [2,2], [1,1], [3,1], [2,1], [3,3], [1,2], [2,3], [3,2]]
X = np.array(X)
print(type(X))

neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

index = list(range(X.shape[0]))
print(neighborhood_centers[1])
distance = 1 / 2 * np.sum((X[1] - neighborhood_centers[1]) ** 2)
print(distance)