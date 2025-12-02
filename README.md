# Interim Assignment 01 - TF-IDF Document Analysis

## Team Members

- Mohammad Marmash (ID: 326791068)
- Mhammad Siag (ID: 213988819)
- Abdullah Abulafi (ID: 213976194)
- Mohammad Ibrahim (ID: 331978114)

## Project Overview

Lightweight TF-IDF engine with keyword filtering and simple dimensionality
reduction visualizations. The project includes a minimal Streamlit UI that
accepts `.txt` files, computes TF-IDF under configurable preprocessing options,
and displays document embeddings using NumPy + Plotly.

## What this repo contains

- `app.py` — Streamlit app (main entrypoint)
- `tfidf.py` — TF-IDF engine (fit/transform, preprocessing options)
- `keyword_vectors.py` — build a reduced keyword matrix from TF-IDF
- `dr.py`, `plots.py` — dimensionality reduction + Plotly visualizations
- sample docs: `test_documents/`
- `requirements.txt` — pinned Python dependencies

## Installation

Create and activate a virtual environment, install dependencies and download
the required NLTK corpora:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Download NLTK resources used by the app
python -m nltk.downloader stopwords wordnet omw-1.4
```

## Running the Application

From the repository root run:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`). Use the
UI to upload `.txt` files and select preprocessing options (lowercase,
remove stopwords, lemmatize) before processing.

## Notes

- Use at least 5 documents for more meaningful visualizations.
- The application assumes UTF-8 encoded text files.
