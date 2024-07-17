# k means with 3d projections but no threshold and does not read from a file

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits import mplot3d


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


# Implementing Expectation step
def assign_clusters(X, clusters):
    for idx in range(X.shape[0]):
        dist = []

        curr_x = X[idx]

        for i in range(k):
            dis = distance(curr_x, clusters[i]['center'])
            dist.append(dis)

        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)
    return clusters


# Implementing the Matching Step
def update_clusters(X, clusters):
    for i in range(k):
        points = np.array(clusters[i]['points'])
        if points.shape[0] > 0:
            new_center = points.mean(axis=0)
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters


def pred_cluster(X, clusters):
    pred = []
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i], clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred


X, y = make_blobs(n_samples=500, n_features=5, centers=100, random_state=23)

k = 5

clusters = {}
np.random.seed(23)

for idx in range(k):
    center = 2 * (2 * np.random.random((X.shape[1],)) - 1)
    points = []
    cluster = {
        'center': center,
        'points': []
    }

    clusters[idx] = cluster

clusters = assign_clusters(X, clusters)
clusters = update_clusters(X, clusters)
pred = pred_cluster(X, clusters)

ax = plt.axes(projection ="3d")

plt.grid(True)
ax.scatter(X[: ,0] ,X[: ,1], X[: ,2], c = pred)
for i in clusters:
    print("\nCluster: " + str(i) + " is centered at " + str(clusters[i]['center']))
    center = clusters[i]['center']
    ax.scatter(center[0] ,center[1] ,center[2] ,marker = '*' ,c = 'red')
plt.show()
