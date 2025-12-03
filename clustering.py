from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

def perform_clustering(df, k):
    X = df.iloc[:, :4]

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_result = kmeans.fit_predict(X)

    silhouette = silhouette_score(X, cluster_result)
    ari = adjusted_rand_score(df["Label_Asli"], cluster_result)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    df["Cluster"] = cluster_result
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]

    return df, silhouette, ari
