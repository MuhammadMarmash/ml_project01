# keywords_repr/keyword_vectors.py

"""
Owner: Sarafande

Goal (MVP):
-----------
Given:
- Full TF-IDF matrix X_full (n_docs, n_vocab)
- feature_names list (vocabulary) in column order
- top_n (how many top words per document)

Produce:
- X_filtered: smaller matrix using only the union of top-N keywords
- keywords_sorted: list of all those keywords, sorted alphabetically
- doc_top_keywords: for each doc, list of (keyword, score) pairs (for the table)

Why:
----
- X_full may be huge. For visualization we only want the most important words.
- The union of top-N keywords across docs reduces dimensionality BEFORE PCA/SVD.
"""

from typing import List, Tuple
import numpy as np


def build_filtered_matrix(
    X_full: np.ndarray,
    feature_names: List[str],
    top_n: int,
) -> tuple[np.ndarray, List[str], List[List[Tuple[str, float]]]]:
    """
    Main function for Part 3 of the assignment.

    Inputs:
    -------
    X_full        : TF-IDF matrix (n_docs, n_vocab)
    feature_names : list of vocabulary terms (len = n_vocab)
    top_n         : how many top keywords per document

    Outputs:
    --------
    X_filtered      : (n_docs, n_keywords_union) using only top keywords
    keywords_sorted : list of all unique keywords from all docs' top-N lists, sorted A-Z
    doc_top_keywords: per-doc list of (keyword, score), sorted by score desc.
    """
    n_docs, n_vocab = X_full.shape

    # ---- Step 1: top-N per document ----
    doc_top_keywords: List[List[Tuple[str, float]]] = []
    union_keywords = set()

    for i in range(n_docs):
        row = X_full[i, :]

        # If row is all zeros (e.g. empty doc), handle gracefully.
        if np.allclose(row, 0.0):
            doc_top_keywords.append([])
            continue

        # Get indices sorted by tf-idf score descending
        # argsort gives ascending, so we reverse.
        sorted_indices = np.argsort(row)[::-1]

        # Take top_n indices (or fewer if many zeros)
        top_indices = [idx for idx in sorted_indices if row[idx] > 0][:top_n]

        keywords_scores: List[Tuple[str, float]] = []
        for idx in top_indices:
            word = feature_names[idx]
            score = float(row[idx])
            keywords_scores.append((word, score))
            union_keywords.add(word)

        # Store for this document (for inspection table)
        doc_top_keywords.append(keywords_scores)

    # ---- Step 2: union of all keywords, sorted alphabetically ----
    keywords_sorted = sorted(union_keywords)

    # Mapping: word -> new column index in filtered matrix
    kw_to_col = {w: j for j, w in enumerate(keywords_sorted)}

    # ---- Step 3: build filtered matrix ----
    n_keywords = len(keywords_sorted)
    X_filtered = np.zeros((n_docs, n_keywords), dtype=np.float64)

    # For each document, we only fill the columns for its own top keywords
    word_to_vocab_idx = {w: i for i, w in enumerate(feature_names)}

    for doc_idx in range(n_docs):
        for (word, score) in doc_top_keywords[doc_idx]:
            # Get column in filtered matrix
            j_filtered = kw_to_col[word]
            # For safety, read score from X_full using original vocab index
            i_vocab = word_to_vocab_idx[word]
            X_filtered[doc_idx, j_filtered] = X_full[doc_idx, i_vocab]

    return X_filtered, keywords_sorted, doc_top_keywords
