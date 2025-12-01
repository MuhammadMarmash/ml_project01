# streamlit_app/app.py

"""
Owner: Siag

Goal (MVP):
-----------
- Build the main Streamlit UI that connects the user to the TFIDF engine.
- Only talk to TFIDF through its public methods (fit, transform, fit_transform, save/load).
- Do NOT do any text preprocessing here (Version 2.0 requirement).
- Store all outputs in st.session_state so other parts (keywords, DR, viz) can reuse them.

Main user flow:
---------------
1. User uploads .txt files.
2. User chooses preprocessing options (checkboxes).
3. User clicks "Process Documents":
   - We read file contents as raw strings.
   - We create TFIDF instance with the checkbox options.
   - We call fit_transform(raw_docs).
   - We store TFIDF object, TF-IDF matrix, file names in session_state.
4. Later steps (keywords, DR, viz) will read from session_state.
"""

from typing import List, Tuple
import io
import pickle

import streamlit as st
import numpy as np

from tfidf import TFIDF
from keywords_repr.keyword_vectors import build_filtered_matrix
from dr import reduce_dimensions
from plots import plot_2d, plot_3d


def _read_uploaded_files(uploaded_files) -> Tuple[List[str], List[str]]:
    """
    Helper function:
    ----------------
    Convert uploaded files to (filenames, contents as strings).

    Why:
    ----
    - TFIDF expects list[str] where each string is the FULL RAW TEXT of a document.
    - Here we only deal with files and encodings, not NLP.
    """
    filenames: List[str] = []
    contents: List[str] = []

    for f in uploaded_files:
        # MVP: assume utf-8, but handle errors gracefully so app doesn't crash.
        try:
            text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            # Fallback encoding if utf-8 fails.
            text = f.read().decode("latin-1", errors="ignore")

        filenames.append(f.name)
        contents.append(text)

    return filenames, contents


def main():
    st.title("TF-IDF Document Analysis (Interim Assignment 01)")

    st.write(
        "Upload text documents, choose preprocessing options, and visualize document relationships."
    )

    # -------- 1. FILE UPLOAD (MVP) --------
    uploaded_files = st.file_uploader(
        "Upload .txt documents",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
    else:
        st.info("Upload at least one .txt file to start.")

    # -------- 2. PREPROCESSING OPTIONS (Version 2.0) --------
    st.subheader("Preprocessing Options")

    lowercase_cb = st.checkbox("Convert to lowercase", value=True)
    stopwords_cb = st.checkbox("Remove stopwords", value=False)
    lemmatize_cb = st.checkbox("Lemmatize", value=False)

    # We can allow user to change top_n as well
    top_n = st.slider("Top-N keywords per document", 3, 30, 10)

    # -------- 3. PROCESS BUTTON: RUN PIPELINE --------
    if st.button("Process Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one file before processing.")
            st.stop()

        filenames, contents = _read_uploaded_files(uploaded_files)

        # Create TFIDF engine with the chosen options
        tfidf = TFIDF(
            lowercase=lowercase_cb,
            remove_stopwords=stopwords_cb,
            lemmatize=lemmatize_cb,
            top_n=top_n,
        )

        # Version 2.0 requirement:
        # We pass RAW texts; preprocessing happens inside the TFIDF class.
        X_full = tfidf.fit_transform(contents)
        feature_names = tfidf.get_feature_names()

        # Store results in session_state so other parts of the app can access them
        st.session_state["filenames"] = filenames
        st.session_state["contents"] = contents
        st.session_state["tfidf"] = tfidf
        st.session_state["X_full"] = X_full
        st.session_state["feature_names"] = feature_names
        st.session_state["top_n"] = top_n

        # Also compute filtered keyword matrix now, for convenience
        X_filtered, keywords_sorted, doc_top_keywords = build_filtered_matrix(
            X_full, feature_names, top_n
        )
        st.session_state["X_filtered"] = X_filtered
        st.session_state["keywords_sorted"] = keywords_sorted
        st.session_state["doc_top_keywords"] = doc_top_keywords

        st.success("Documents processed successfully. Scroll down for visualization.")

    # If we don't have processed data yet, we stop here.
    if "X_filtered" not in st.session_state:
        return

    # Short aliases
    filenames: List[str] = st.session_state["filenames"]
    X_filtered: np.ndarray = st.session_state["X_filtered"]
    doc_top_keywords = st.session_state["doc_top_keywords"]
    top_n: int = st.session_state["top_n"]

    st.subheader("Dimensionality Reduction Options")

    # -------- DR CONFIG (for Abdallah's code) --------
    method = st.radio("Reduction method", ["PCA", "SVD"], index=0)
    dim_choice = st.radio("Number of dimensions", ["2D", "3D"], index=0)
    n_components = 2 if dim_choice == "2D" else 3

    coords = reduce_dimensions(
        X_filtered,
        method=method,
        n_components=n_components,
    )
    st.session_state["coords"] = coords

    # -------- DOCUMENT INSPECTION (for Sarafande's code) --------
    st.subheader("Inspect Individual Document")

    options = ["(None)"] + filenames
    selected_name = st.selectbox("Select a document", options, index=0)

    if selected_name != "(None)":
        selected_idx = filenames.index(selected_name)
    else:
        selected_idx = None

    if selected_idx is not None:
        st.markdown(f"**Top {top_n} keywords for:** `{selected_name}`")
        doc_keywords = doc_top_keywords[selected_idx]  # list[(keyword, score)]

        import pandas as pd

        df_kw = pd.DataFrame(doc_keywords, columns=["Keyword", "TF-IDF Score"])
        df_kw = df_kw.sort_values("TF-IDF Score", ascending=False)
        st.dataframe(df_kw, use_container_width=True)

    # -------- PLOTTING (for Abdallah's code) --------
    st.subheader("Document Embedding Visualization")

    if coords.shape[1] == 2:
        fig = plot_2d(coords, filenames, selected_idx)
    else:
        fig = plot_3d(coords, filenames, selected_idx)

    st.plotly_chart(fig, use_container_width=True)

    # -------- ANALYSIS TEXT AREA --------
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

    # -------- SAVE MODEL (MVP) --------
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
