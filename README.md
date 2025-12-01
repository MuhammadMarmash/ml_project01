# TF-IDF Document Analysis (Project 01)

Small teaching project that implements a lightweight TF-IDF engine, keyword
filtering and simple dimensionality-reduction visualizations using NumPy and
Plotly. The project is wired to a minimal Streamlit UI for interactive use.

## What this repo contains

- `app.py` — Streamlit app (main entrypoint)
- `tfidf.py` — simple TF-IDF engine (fit/transform, preprocessing options)
- `keyword_vectors.py` — build a reduced keyword matrix from TF-IDF
- `dr.py`, `plots.py` — dimensionality reduction + Plotly visualizations
- sample docs: `doc1.txt`, `doc2.txt`, `doc3.txt`
- `requirements.txt` — pinned Python dependencies
- `tests/` — pytest unit tests used by the course

## Quickstart (local)

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required NLTK data (stopwords + wordnet):

```bash
python -m nltk.downloader stopwords wordnet
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

The app UI lets you upload `.txt` documents, choose preprocessing options
(lowercase, remove stopwords, lemmatize), compute TF-IDF and visualize
document embeddings.

## Tests

Run the unit tests with:

```bash
pytest -q
```

## Notes

- This project intentionally avoids scikit-learn for the core TF-IDF and PCA
  exercises; NumPy is used instead to keep implementations explicit.
- If you plan to publish the repository, make sure your local `.venv` and any
  large data files are excluded (see `.gitignore`).

## Remote

This repository can be pushed to `git@github.com:MuhammadMarmash/ml_project01.git`.
