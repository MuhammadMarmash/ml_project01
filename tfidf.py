"""
TF-IDF engine (single-file implementation expected by the app)

This module provides the `TFIDF` class implementing Version 2.0 requirements:
- preprocessing options in `__init__`
- preprocessing happens inside `fit()` and `transform()`
- `fit()`, `transform()`, `fit_transform()`
- `get_feature_names()`, `save_to_file()`, `load_from_file()`

The implementation mirrors the working version previously in `core_tfidf/tfidf_engine.py`.
"""

from typing import List, Dict, Optional
import re
import pickle

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TFIDF:
    """
    Simple TF-IDF implementation with configurable preprocessing.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        lemmatize: bool = False,
        top_n: int = 10,
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.top_n = top_n

        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Optional[np.ndarray] = None
        self.N_: Optional[int] = None

        self._stopwords: Optional[set[str]] = None
        self._lemmatizer: Optional[WordNetLemmatizer] = None

        if self.remove_stopwords:
            self._stopwords = set(stopwords.words("english"))

        if self.lemmatize:
            self._lemmatizer = WordNetLemmatizer()

    def _preprocess(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()

        tokens = re.findall(r"\b\w+\b", text)

        if self.remove_stopwords and self._stopwords is not None:
            tokens = [t for t in tokens if t not in self._stopwords]

        if self.lemmatize and self._lemmatizer is not None:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return tokens

    def fit(self, documents: List[str]) -> "TFIDF":
        N = len(documents)
        self.N_ = N

        doc_word_sets: List[set[str]] = []
        for doc in documents:
            tokens = self._preprocess(doc)
            doc_word_sets.append(set(tokens))

        all_words = set()
        for s in doc_word_sets:
            all_words.update(s)

        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(all_words))}

        df = np.zeros(len(self.vocabulary_), dtype=np.int64)
        for word, idx in self.vocabulary_.items():
            df[idx] = sum(1 for s in doc_word_sets if word in s)

        self.idf_ = np.log((N + 1) / (df + 1))
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        if self.idf_ is None or not self.vocabulary_:
            raise ValueError("TFIDF model is not fitted yet. Call fit() first.")

        n_docs = len(documents)
        n_vocab = len(self.vocabulary_)
        X = np.zeros((n_docs, n_vocab), dtype=np.float64)

        for i, doc in enumerate(documents):
            tokens = self._preprocess(doc)
            if not tokens:
                continue

            total_words = len(tokens)
            counts: Dict[int, int] = {}
            for token in tokens:
                if token in self.vocabulary_:
                    j = self.vocabulary_[token]
                    counts[j] = counts.get(j, 0) + 1

            for j, c in counts.items():
                tf = c / total_words
                X[i, j] = tf * self.idf_[j]

        return X

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        idx_to_word = sorted(self.vocabulary_.items(), key=lambda x: x[1])
        return [w for (w, idx) in idx_to_word]

    def save_to_file(self, filepath: str) -> None:
        data = {
            "lowercase": self.lowercase,
            "remove_stopwords": self.remove_stopwords,
            "lemmatize": self.lemmatize,
            "top_n": self.top_n,
            "vocabulary_": self.vocabulary_,
            "idf_": self.idf_,
            "N_": self.N_,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load_from_file(cls, filepath: str) -> "TFIDF":
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            lowercase=data["lowercase"],
            remove_stopwords=data["remove_stopwords"],
            lemmatize=data["lemmatize"],
            top_n=data.get("top_n", 10),
        )
        obj.vocabulary_ = data["vocabulary_"]
        obj.idf_ = data["idf_"]
        obj.N_ = data["N_"]
        return obj
