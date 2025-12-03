import streamlit as st
import matplotlib.pyplot as plt

from data_loader import load_data
from clustering import perform_clustering

# Judul Aplikasi
st.set_page_config(page_title="Clustering Iris", layout="centered")
st.title("Perbandingan Clustering K-Means dan Label Asli Iris")

# Load Dataset
df, target_names = load_data()
st.subheader("Dataset Iris")
st.dataframe(df)

# Input Jumlah Cluster
k = st.slider("Pilih Jumlah Cluster (K)", 2, 6, 3)

# Proses Clustering
df, silhouette, ari = perform_clustering(df, k)

# Evaluasi
st.subheader("Evaluasi")
st.write(f"Silhouette Score: {silhouette:.3f}")
st.write(f"Adjusted Rand Index (ARI): {ari:.3f}")

# Visualisasi Clustering
st.subheader("Hasil Clustering")

fig1, ax1 = plt.subplots()
ax1.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"])
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.set_title("Clustering K-Means")
st.pyplot(fig1)

# Visualisasi Label Asli
st.subheader("Label Asli")

fig2, ax2 = plt.subplots()
ax2.scatter(df["PCA1"], df["PCA2"], c=df["Label_Asli"])
ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.set_title("Label Asli Iris")
st.pyplot(fig2)

# Tabel Perbandingan
st.subheader("Perbandingan Data")
st.dataframe(df[[
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "Label_Asli",
    "Cluster"
]])

# Kesimpulan
st.subheader("Kesimpulan")
st.write(
    "Clustering mengelompokkan data berdasarkan kemiripan fitur, "
    "sedangkan label asli merupakan pengelompokan biologis. "
    "Nilai ARI menunjukkan tingkat kesesuaian hasil clustering."
)
