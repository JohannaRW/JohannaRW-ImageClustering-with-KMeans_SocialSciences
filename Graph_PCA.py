import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter


def plot_clusters_pca(X, labels, kmeans_model=None, title="Clusters in PCA space", random_state=42):
    """
    Plot a 2D visualization of clusters in PCA space.
    Adds cluster numbers and sizes (n=...).

    Parameters
    ----------
    X : ndarray
        Feature matrix (e.g., PCA-reduced features or original features).
    labels : ndarray
        Cluster labels for each sample.
    kmeans_model : KMeans, optional
        If provided, cluster centers will be plotted.
    title : str
        Plot title.
    random_state : int
        Reproducibility for PCA.
    """

    # Reduce to 2D with PCA
    reducer = PCA(n_components=2, random_state=random_state)
    X_2d = reducer.fit_transform(X)

    # Cluster sizes
    counts = Counter(labels)

    # Define custom color palette (blue, green, yellow, orange, red, purple, etc.)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b",
              "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    plt.figure(figsize=(10, 8))
    for cl in sorted(set(labels)):
        idxs = np.where(labels == cl)[0]
        plt.scatter(
            X_2d[idxs, 0],
            X_2d[idxs, 1],
            c=colors[cl % len(colors)],
            s=30,
            alpha=0.7,
            label=f"Cluster {cl} (n={counts[cl]})"
        )

    # Plot cluster centers if KMeans is given
    if kmeans_model is not None:
        centers_2d = reducer.transform(kmeans_model.cluster_centers_)
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c="black",
            s=200,
            alpha=0.9,
            marker="X",
            label="Cluster centers"
        )

    plt.title(title)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.legend()
    plt.tight_layout()
    plt.show()
