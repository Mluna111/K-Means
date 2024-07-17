#K_means with 2D projections, reads from file, and has threshold to stop

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

def distance(p1,p2):
    return np.sqrt(np.sum((p1-p2)**2))


#Implementing Expecation step
def assign_clusters(X, clusters):
  #Assigns each point in X to the cluster closest (smallest mean
    for idx in range(X.shape[0]):
        dist = []

        curr_x = X[idx]

        #checks distance to every cluster
        for i in range(k):
            dis = distance(curr_x,clusters[i]['center'])
            dist.append(dis)

        #finds closest cluster and adds point to it
        curr_cluster = np.argmin(dist)
        clusters[curr_cluster]['points'].append(curr_x)

    return clusters

#Implementing the Matching Step
def update_clusters(X, clusters):
  difference = 0;
  #updates the center of each cluster
  for i in range(k):
        points = np.array(clusters[i]['points'])

        #finds mean of points in cluster
        if points.shape[0] > 0:
            new_center = points.mean(axis =0)
            difference += distance(new_center, clusters[i]['center'])
            clusters[i]['center'] = new_center
            clusters[i]['points'] = []
  return clusters, difference


#Calculate distance again for new centers of each cluster and assign them to a cluster in the prediction
def pred_cluster(X, clusters):
    pred = []

    #finds the cluster closest to each point
    for i in range(X.shape[0]):
        dist = []
        for j in range(k):
            dist.append(distance(X[i],clusters[j]['center']))
        pred.append(np.argmin(dist))
    return pred

#Generating data
#X,y = make_blobs(n_samples = 10000,n_features = 3, centers = 50,random_state = 23)

#load data from file
    # 0) Assessment Level (x10) | 1) Attempts on Post (x10) | 2) Pre Assessment Score | 3) Post Assessment Score | 4) Chapters Mastered (x100) | 5) Improvement/Session (÷10) | 6) Attendance Count (x100)
X = np.loadtxt('Test1', dtype=float)
k = 5

clusters = {}
seed = 28
np.random.seed(seed)

#initializes the clusters
for idx in range(k):
    #center = np.random.random((X.shape[1],))
    center = X[np.random.randint(X.shape[0])]
    #int(np.random.random() * len(X))
    points = []
    cluster = {
        'center' : center,
        'points' : []
    }

    clusters[idx] = cluster
print(clusters)

difference = 10000
while(difference > 0.0001):
    clusters = assign_clusters(X,clusters)
    clusters, difference = update_clusters(X,clusters)

pred = pred_cluster(X,clusters)
np.savetxt('pred1.txt', pred, fmt='%d')

fig = plt.figure(figsize = (8,6))

#plt.xlim(-0.001, 1)
plt.ylim(-0.02, 1)

plt.xlabel('Assessment Level (x10)')
plt.ylabel('Attendance Count (x100)')
plt.title('Assessment Level vs Attendance Count')
plt.text(0,0, "K: "+str(k)+" S: "+ str(seed))


plt.grid(True)
plt.scatter(X[:,0], X[:,6], c = pred)
for i in clusters:
    center = clusters[i]['center']
    plt.scatter(center[0],center[6],marker = 'p',c = 'red')
plt.show()
