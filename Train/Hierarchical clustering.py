import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=300,
    centers=3,
    random_state=42
)

model = KMeans(
    n_clusters=3,
    random_state=42
)

model.fit(X)

labels = model.labels_

plt.scatter(
    X[:,0],
    X[:,1],
    c=labels
)

plt.show()
