# Unsupervised Learning Example: Cluster Iris Flowers
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()
X = iris.data  # Only features, no labels used

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Visualize clusters (2D using only two features)
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title("K-Means Clustering (Unsupervised)")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()
