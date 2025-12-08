# Edge Cases Handling

This document outlines how the TF-IDF pipeline handles edge cases gracefully.

## 1. Empty Documents After Preprocessing ✅

**Scenario**: User uploads a document that becomes empty after preprocessing (e.g., only stopwords).

**How it's handled**:

- `tfidf.py` line 61: `if not tokens: continue` — skips IDF computation for empty docs
- `app.py`: No longer filters out empty documents; they are kept with all-zero vectors
- `keyword_vectors.py`: Takes top-N keywords even if document has no significant scores (gets top-N zeros)
- `dr.py`: PCA/SVD handles zero vectors naturally (they cluster near origin)
- `plots.py`: Renders zero-vector documents at or near the origin

**Result**: Empty documents don't crash the app; they appear at (0,0) or (0,0,0) on plots.

---

## 2. Word Appears in All Documents (IDF → 0) ✅

**Scenario**: A word appears in every document in the training set.

**How it's handled**:

- `tfidf.py` line 65: `df[idx] = sum(1 for s in doc_word_sets if word in s)`
- `tfidf.py` line 67: `self.idf_ = np.log((N + 1) / (df + 1))`
  - When `df == N`: `IDF = log((N+1)/(N+1)) = log(1) = 0`
  - This is **mathematically correct** — words in all docs have no discriminative power
- `tfidf.py` line 83: `X[i, j] = tf * self.idf_[j]` — results in 0 TF-IDF score for ubiquitous words

**Result**: Ubiquitous words are naturally down-weighted to zero; no special handling needed.

---

## 3. Unknown Words in Transform() ✅

**Scenario**: User calls `transform()` on documents containing words not seen during `fit()`.

**How it's handled**:

- `tfidf.py` line 78-81:
  ```python
  if token in self.vocabulary_:
      j = self.vocabulary_[token]
      counts[j] = counts.get(j, 0) + 1
  ```
- Unknown words are **silently ignored** (not added to counts)
- Only words in the fitted vocabulary contribute to TF-IDF

**Result**: New words are handled gracefully without crashes or warnings.

---

## 4. File Reading Errors (Encoding) ✅

**Scenario**: Uploaded file has encoding issues or is corrupted.

**How it's handled**:

- `app.py` lines 22-26:
  ```python
  for f in uploaded_files:
      try:
          text = f.read().decode("utf-8", errors="ignore")
      except Exception:
          text = f.read().decode("latin-1", errors="ignore")
  ```
- Attempts UTF-8 first, falls back to latin-1
- `errors="ignore"` skips unencodable bytes instead of crashing
- All files are successfully read regardless of encoding

**Result**: Files with mixed or unusual encodings are processed without errors.

---

## 5. Empty Filtered Matrix (No Keywords) ⚠️

**Scenario**: All documents have zero TF-IDF vectors (e.g., all stopwords).

**How it's handled**:

- `keyword_vectors.py` lines 26-28:
  ```python
  for idx, score in top_pairs:
      word = feature_names[idx]
      ...union_keywords.add(word)
  ```
- Even zero-value words are added to union (top-N list is always length N, even with zeros)
- Result: `keywords_sorted` may contain zero-weight words, `X_filtered` is mostly zeros
- `dr.py` line 33: `if X.size == 0:` returns zero matrix (handles empty feature case)
- `plots.py`: Displays all documents clustered at origin

**Result**: Graceful degradation; app continues to work with zero-vector documents.

---

## 6. Single Document ⚠️

**Scenario**: User uploads only 1 document.

**How it's handled**:

- `tfidf.py` line 67: `N = 1`
- `IDF = log((1+1)/(df+1))`
- When word appears: `df = 1`, so `IDF = log(2/2) = 0`
- **All TF-IDF scores become 0** (every word has IDF = 0)

**Known limitation**: PCA/SVD requires at least 2 documents to compute meaningful embeddings.

- With 1 doc: All coordinates are zero
- With 2 docs: PCA/SVD work, one doc has positive score, one is mirrored negative

**Recommendation**: Ask user to upload ≥2 documents for meaningful analysis.

---

## 7. Empty Vocabulary ✅

**Scenario**: Documents contain no valid tokens after preprocessing.

**How it's handled**:

- `tfidf.py` line 47: `self.vocabulary_ = {}` (empty dict)
- `tfidf.py` line 54: `if n_vocab == 0: return np.zeros((n_docs, 0), dtype=np.float64)`
- `keyword_vectors.py` line 21: `top_pairs = sorted_scores[:top_n]` — takes first 10 (but list is empty)
- `keyword_vectors.py` line 26: `if not doc_top_keywords:` loop is skipped
- `keyword_vectors.py` line 39: `X_filtered = np.zeros((n_docs, 0))`

**Result**: Empty vocabulary produces empty feature matrix; app shows message and continues gracefully.

---

## Summary Table

| Edge Case        | Handled? | Behavior                     | Impact                |
| ---------------- | -------- | ---------------------------- | --------------------- |
| Empty documents  | ✅       | Kept, appear at origin       | None                  |
| Word in all docs | ✅       | IDF → 0, natural             | None                  |
| Unknown words    | ✅       | Silently ignored             | None                  |
| Encoding errors  | ✅       | UTF-8 → latin-1 fallback     | None                  |
| No keywords      | ⚠️       | Empty matrix, zero vectors   | Degraded but works    |
| Single document  | ⚠️       | All IDF → 0, all coords zero | Limited utility       |
| Empty vocabulary | ✅       | Zero matrix, continues       | Works but no features |
