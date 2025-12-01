import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_plot(X_scaled, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="viridis")

    plt.colorbar(scatter, ax=ax)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Visualization of Clusters")

    plt.tight_layout()
    return fig
