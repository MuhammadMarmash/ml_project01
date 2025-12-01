# test_keywords_repr.py
# Goal: verify top-N keyword union + filtered matrix logic.
import sys
import pathlib
import os

# Ensure project root is on sys.path so this file can be run directly
ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tfidf import TFIDF
from keyword_vectors import build_filtered_matrix
import numpy as np

docs = [
    "pizza love cheese italian food",
    "python programming language code software",
    "python snake nature reptile wildlife",
]

def main():
    tfidf = TFIDF(lowercase=True, remove_stopwords=False, lemmatize=False, top_n=3)
    X_full = tfidf.fit_transform(docs)
    feature_names = tfidf.get_feature_names()

    print("Full TF-IDF shape:", X_full.shape)
    print("Feature names:", feature_names)

    X_filtered, keywords_sorted, doc_top_keywords = build_filtered_matrix(
        X_full, feature_names, top_n=3
    )

    print("Filtered TF-IDF shape:", X_filtered.shape)
    print("Filtered keywords:", keywords_sorted)
    print("Doc top keywords:")
    for i, kw_list in enumerate(doc_top_keywords):
        print(f"Doc {i}:", kw_list)

    # Sanity checks
    assert X_filtered.shape[0] == len(docs)
    assert len(keywords_sorted) <= 3 * len(docs)

    print("✅ keywords_repr tests passed OK!")

if __name__ == "__main__":
    main()
