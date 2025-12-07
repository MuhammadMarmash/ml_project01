from typing import List, Tuple
import numpy as np


def build_filtered_matrix(
    X_full: np.ndarray,
    feature_names: List[str],
    top_n: int,
) -> tuple[np.ndarray, List[str], List[List[Tuple[str, float]]]]:
    n_docs, n_vocab = X_full.shape

    doc_top_keywords: List[List[Tuple[str, float]]] = []
    union_keywords = set()

    for i in range(n_docs):
        row = X_full[i, :]

        if np.allclose(row, 0.0):
            doc_top_keywords.append([])
            continue

        sorted_indices = np.argsort(row)[::-1]

        top_indices = [idx for idx in sorted_indices if row[idx] > 0][:top_n]

        keywords_scores: List[Tuple[str, float]] = []
        for idx in top_indices:
            word = feature_names[idx]
            score = float(row[idx])
            keywords_scores.append((word, score))
            union_keywords.add(word)

        doc_top_keywords.append(keywords_scores)

    keywords_sorted = sorted(union_keywords)

    kw_to_col = {w: j for j, w in enumerate(keywords_sorted)}

    n_keywords = len(keywords_sorted)
    X_filtered = np.zeros((n_docs, n_keywords), dtype=np.float64)

    word_to_vocab_idx = {w: i for i, w in enumerate(feature_names)}

    for doc_idx in range(n_docs):
        for (word, score) in doc_top_keywords[doc_idx]:
            j_filtered = kw_to_col[word]
            i_vocab = word_to_vocab_idx[word]
            X_filtered[doc_idx, j_filtered] = X_full[doc_idx, i_vocab]

    return X_filtered, keywords_sorted, doc_top_keywords
