import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

from clustering import run_kmeans, evaluate_clusters
from visualization import pca_plot


st.set_page_config(page_title="Iris Clustering SPK", page_icon="🌸", layout="wide")

st.title("Sistem Pendukung Keputusan – Iris Clustering")
st.write("Versi simpel & estetik dengan 4 file saja.")

uploaded_file = st.file_uploader("Upload Dataset Iris (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil di-upload!")

    st.write("### Preview Dataset")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # PARAMETER
    k = st.slider("Jumlah cluster (k)", 2, 10, 3)
    do_scale = st.checkbox("Gunakan StandardScaler", value=True)

    if st.button("Jalankan Clustering"):
        X = df.iloc[:, :-1]

        if do_scale:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values

        labels = run_kmeans(X_scaled, k)
        df["cluster"] = labels

        st.success("Clustering selesai!")
        st.dataframe(df)

        # Evaluasi
        sil, ch, db = evaluate_clusters(X_scaled, labels)

        st.markdown("Evaluasi Model")
        st.write(f"**Silhouette Score:** {sil:.4f}")
        st.write(f"**Calinski-Harabasz:** {ch:.2f}")
        st.write(f"**Davies-Bouldin:** {db:.4f}")

        st.markdown("---")

        # Visualisasi
        st.markdown("Visualisasi Klaster (PCA)")
        fig = pca_plot(X_scaled, labels)
        st.pyplot(fig)
