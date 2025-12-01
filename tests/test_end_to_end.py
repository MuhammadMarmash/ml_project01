import os
import sys
import pathlib
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tfidf import TFIDF
from keyword_vectors import build_filtered_matrix
from dr import reduce_dimensions
from plots import plot_2d, plot_3d
import numpy as np
import plotly.graph_objs as go


def read_docs():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    paths = [os.path.join(root, f"doc{i}.txt") for i in (1, 2, 3)]
    docs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs


def test_full_pipeline(tmp_path):
    docs = read_docs()
    assert len(docs) == 3

    # Create model with preprocessing enabled (exercise stopwords + lemmatizer)
    model = TFIDF(lowercase=True, remove_stopwords=True, lemmatize=True, top_n=5)

    # Fit and transform
    X_full = model.fit_transform(docs)

    # Basic checks
    feature_names = model.get_feature_names()
    assert X_full.shape[0] == 3
    assert len(feature_names) == X_full.shape[1]

    # Build filtered matrix
    X_filtered, keywords_sorted, doc_top_keywords = build_filtered_matrix(
        X_full, feature_names, model.top_n
    )

    # Filtered matrix should have same number of rows
    assert X_filtered.shape[0] == 3
    # keywords_sorted length should match filtered matrix columns
    assert len(keywords_sorted) == X_filtered.shape[1]

    # Check that doc_top_keywords length equals docs
    assert len(doc_top_keywords) == 3

    # Dimensionality reduction 2D and 3D
    coords2 = reduce_dimensions(X_filtered, method="PCA", n_components=2)
    assert coords2.shape == (3, 2)

    coords3 = reduce_dimensions(X_filtered, method="SVD", n_components=3)
    assert coords3.shape == (3, 3)

    # Plot functions return plotly figures
    fig2 = plot_2d(coords2, ["d1", "d2", "d3"], selected_idx=None)
    fig3 = plot_3d(coords3, ["d1", "d2", "d3"], selected_idx=1)
    assert isinstance(fig2, go.Figure)
    assert isinstance(fig3, go.Figure)

    # Test save/load
    p = tmp_path / "model.pkl"
    model.save_to_file(str(p))
    loaded = TFIDF.load_from_file(str(p))
    assert loaded.lowercase == model.lowercase
    assert loaded.remove_stopwords == model.remove_stopwords
    assert loaded.lemmatize == model.lemmatize
    assert loaded.top_n == model.top_n
    # Vocabulary should match
    assert loaded.vocabulary_ == model.vocabulary_
