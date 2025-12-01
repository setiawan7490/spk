import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def pca_plot(X_scaled, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")

    plt.colorbar(scatter)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization of Clusters")

    return fig
