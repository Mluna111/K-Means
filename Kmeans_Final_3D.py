
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

# Euclidian distance
def distance(p1 ,p2):
    return np.sqrt(np.sum((p1 - p2)**2))

# Implementing assign step or expectation step
def assign_clusters(X, clusters):
    # Assigns each point in X to the cluster closest (smallest mean
    for idx in range(X.shape[0]):
        dist = []

        curr_x = X[idx]

        # checks distance to every cluster
        for i in range(k):
            dis = distance(curr_x ,clusters[i]['center'])
            dist.append(dis)

        # finds closest cluster and adds point to it
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)

    return clusters

# Implementing the update step or matching step
def update_clusters(X, clusters):
    difference = 0;
    # updates the center of each cluster
    for i in range(k):
        points = np.array(clusters[i]['points'])

        # finds mean of points in cluster
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            difference += distance(new_center, clusters[i]['center'])
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
    return clusters, difference

# Final assign step
def pred_cluster(X, clusters):
    pred = []

    # finds the cluster closest to each point
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i] ,clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred


# Generating random data
# X,y = make_blobs(n_samples = 10000,n_features = 3, centers = 50,random_state = 23)

# loading data from file
    # 0) Assessment Level (x10) | 1) Attempts on Post (x10) | 2) Pre Assessment Score | 3) Post Assessment Score | 4) Chapters Mastered (x100) | 5) Improvement/Session (÷10) | 6) Attendance Count (x100)
X = np.loadtxt('Test1', dtype=float)
k = 5

clusters = {}
seed = 28
np.random.seed(seed)

# initializes the clusters
for idx in range(k):
    # center = np.random.random((X.shape[1],))
    center = X[np.random.randint(X.shape[0])]
    # int(np.random.random() * len(X))
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }

    clusters[idx] = cluster

difference = 10000
while(difference > 0.0001):
    clusters = assign_clusters(X ,clusters)
    clusters, difference = update_clusters(X ,clusters)

pred = pred_cluster(X ,clusters)

np.savetxt('pred1.txt', pred, fmt='%d')

ax = plt.axes(projection ="3d")

plt.ylim(-0.001, 1)
plt.xlim(-0.001, .6)

ax.set_xlabel('Improvement/Session (÷10)')
ax.set_ylabel('Attendance Count (x100)')
ax.set_zlabel('Assessment Level (x10)')

plt.grid(True)
ax.scatter(X[: ,5] ,X[: ,6], X[: ,0], c = pred)
for i in clusters:
    print("\nCluster: " + str(i) + " is centered at " + str(clusters[i]['center']))
    center = clusters[i]['center']
    ax.scatter(center[5] ,center[6] ,center[0] ,marker = '*' ,c = 'red')
plt.show()
