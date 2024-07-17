import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.loadtxt('Test1', dtype=float)

# Find optimum number of cluster
sse = []  # SUM OF SQUARED ERROR
for k in range(1, 15):
    km = KMeans(n_clusters=k, random_state=3)
    km.fit(X)
    sse.append(km.inertia_)

sns.set_style("whitegrid")
g = sns.lineplot(x=range(1, 15), y=sse)

g.set(xlabel="Number of cluster (k)",
      ylabel="Sum Squared Error",
      title='Elbow Method on Set 1')

plt.show()

