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

    # 1) Same as your friend's: compute top_n keywords per doc
    for doc_idx in range(n_docs):
        row = X_full[doc_idx, :]

        index_scores = list(enumerate(row))
        sorted_scores = sorted(index_scores, key=lambda t: t[1], reverse=True)
        top_pairs = sorted_scores[:top_n]

        keywords_scores: List[Tuple[str, float]] = []
        for idx, score in top_pairs:
            word = feature_names[idx]
            score = float(score)
            keywords_scores.append((word, score))
            union_keywords.add(word)

        doc_top_keywords.append(keywords_scores)

    # 2) Same union + sorted list as friend
    keywords_sorted = sorted(union_keywords)

    # 3) Map word -> original vocab index
    word_to_vocab_idx = {w: i for i, w in enumerate(feature_names)}

    # 4) Build filtered matrix EXACTLY like his code
    n_keywords = len(keywords_sorted)
    X_filtered = np.zeros((n_docs, n_keywords), dtype=np.float64)

    for i in range(n_docs):
        for j, word in enumerate(keywords_sorted):
            vocab_idx = word_to_vocab_idx[word]
            X_filtered[i, j] = X_full[i, vocab_idx]

    return X_filtered, keywords_sorted, doc_top_keywords
