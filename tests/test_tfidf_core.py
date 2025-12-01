# test_tfidf_core.py
# Goal: make sure TFIDF itself works before touching Streamlit.

import sys
import pathlib
import os

# Ensure project root is on sys.path so this file can be run directly
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tfidf import TFIDF
import numpy as np

docs = [
    "Machine learning is a subset of artificial intelligence. Machine learning algorithms learn from data.",
    "Deep learning is a subset of machine learning. Deep learning uses neural networks.",
    "Natural language processing is a field of artificial intelligence. Natural language processing deals with text data.",
]

def main():
    # 1) Create model with simple options
    tfidf = TFIDF(
        lowercase=True,
        remove_stopwords=False,
        lemmatize=False,
        top_n=5,
    )

    # 2) Fit + transform
    X = tfidf.fit_transform(docs)
    vocab = tfidf.get_feature_names()

    print("Vocabulary size:", len(vocab))
    print("Vocabulary sample:", vocab[:20])
    print("TF-IDF matrix shape:", X.shape)
    print("TF-IDF matrix (rounded):")
    print(np.round(X, 3))

    # 3) Basic sanity checks
    assert X.shape[0] == len(docs), "Rows should equal number of documents"
    assert X.shape[1] == len(vocab), "Columns should equal vocabulary size"

    # 4) Transform new documents (make sure transform() works)
    new_docs = ["Machine learning and deep learning are related fields."]
    X_new = tfidf.transform(new_docs)
    print("New doc TF-IDF shape:", X_new.shape)
    assert X_new.shape[1] == len(vocab), "New docs must use same vocabulary"

    # 5) Test save + load
    filepath = "tmp_tfidf_model.pkl"
    tfidf.save_to_file(filepath)
    loaded = TFIDF.load_from_file(filepath)

    X_loaded = loaded.transform(docs)
    assert np.allclose(X, X_loaded), "Loaded model should produce same TF-IDF values"

    print("✅ core_tfidf tests passed OK!")

if __name__ == "__main__":
    main()
