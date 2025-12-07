from typing import List, Tuple
import io
import pickle

import streamlit as st
import numpy as np

from tfidf import TFIDF
from keyword_vectors import build_filtered_matrix
from dr import reduce_dimensions
from plots import plot_2d, plot_3d


def _read_uploaded_files(uploaded_files) -> Tuple[List[str], List[str]]:
    filenames: List[str] = []
    contents: List[str] = []

    for f in uploaded_files:
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            text = f.read().decode("latin-1", errors="ignore")

        filenames.append(f.name)
        contents.append(text)

    return filenames, contents


def main():
    st.title("TF-IDF Document Analysis (Interim Assignment 01)")

    st.write(
        "Upload text documents, choose preprocessing options, and visualize document relationships."
    )

    uploaded_files = st.file_uploader(
        "Upload .txt documents",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
    else:
        st.info("Upload at least one .txt file to start.")

    st.subheader("Preprocessing Options")

    lowercase_cb = st.checkbox("Convert to lowercase", value=True)
    stopwords_cb = st.checkbox("Remove stopwords", value=False)
    lemmatize_cb = st.checkbox("Lemmatize", value=False)

    top_n = st.slider("Top-N keywords per document", 3, 30, 10)

    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one file before processing.")
            st.stop()

        filenames, contents = _read_uploaded_files(uploaded_files)

        tfidf = TFIDF(
            lowercase=lowercase_cb,
            remove_stopwords=stopwords_cb,
            lemmatize=lemmatize_cb,
            top_n=top_n,
        )

        X_full = tfidf.fit_transform(contents)
        feature_names = tfidf.get_feature_names()

        st.session_state["filenames"] = filenames
        st.session_state["contents"] = contents
        st.session_state["tfidf"] = tfidf
        st.session_state["X_full"] = X_full
        st.session_state["feature_names"] = feature_names
        st.session_state["top_n"] = top_n

        X_filtered, keywords_sorted, doc_top_keywords = build_filtered_matrix(
            X_full, feature_names, top_n
        )

        non_empty_indices = []
        for i in range(X_filtered.shape[0]):
            if not np.allclose(X_filtered[i, :], 0.0):
                non_empty_indices.append(i)

        X_filtered = X_filtered[non_empty_indices, :]
        filenames_filtered = [filenames[i] for i in non_empty_indices]
        doc_top_keywords_filtered = [doc_top_keywords[i] for i in non_empty_indices]

        st.session_state["X_filtered"] = X_filtered
        st.session_state["keywords_sorted"] = keywords_sorted
        st.session_state["doc_top_keywords"] = doc_top_keywords_filtered
        st.session_state["filenames"] = filenames_filtered

        st.success("Documents processed successfully. Scroll down for visualization.")

    if "X_filtered" not in st.session_state:
        return

    filenames: List[str] = st.session_state["filenames"]
    X_filtered: np.ndarray = st.session_state["X_filtered"]
    doc_top_keywords = st.session_state["doc_top_keywords"]
    top_n: int = st.session_state["top_n"]

    st.subheader("Dimensionality Reduction Options")

    method = st.radio("Reduction method", ["PCA", "SVD"], index=0)
    dim_choice = st.radio("Number of dimensions", ["2D", "3D"], index=0)
    n_components = 2 if dim_choice == "2D" else 3

    coords = reduce_dimensions(
        X_filtered,
        method=method,
        n_components=n_components,
    )
    st.session_state["coords"] = coords

    st.subheader("Inspect Individual Document")

    options = ["(None)"] + filenames
    selected_name = st.selectbox("Select a document", options, index=0)

    if selected_name != "(None)":
        selected_idx = filenames.index(selected_name)
    else:
        selected_idx = None

    if selected_idx is not None:
        st.markdown(f"**Top {top_n} keywords for:** `{selected_name}`")
        doc_keywords = doc_top_keywords[selected_idx]

        import pandas as pd

        df_kw = pd.DataFrame(doc_keywords, columns=["Keyword", "TF-IDF Score"])
        df_kw = df_kw.sort_values("TF-IDF Score", ascending=False)
        st.dataframe(df_kw, use_container_width=True)

    st.subheader("Document Embedding Visualization")

    if coords.shape[1] == 2:
        fig = plot_2d(coords, filenames, selected_idx)
    else:
        fig = plot_3d(coords, filenames, selected_idx)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Analysis (Your Notes)")

    st.markdown(
        """
Use this area to write your observations (for the exam):
- Do you see clusters of similar documents?
- How do preprocessing options change the positions?
- Differences between PCA and SVD?
- Do top keywords explain the clusters?
"""
    )
    _ = st.text_area("Your analysis (not saved, just for practice):", height=150)

    st.subheader("Save Trained TF-IDF Model")

    tfidf = st.session_state["tfidf"]
    buffer = io.BytesIO()
    pickle.dump(tfidf, buffer)
    buffer.seek(0)

    st.download_button(
        "Download TF-IDF model (.pkl)",
        data=buffer,
        file_name="tfidf_model.pkl",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
