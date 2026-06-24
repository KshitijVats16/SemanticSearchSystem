# Semantic Search System

A semantic search engine built using Machine Learning, Natural Language Processing, and FastAPI.

## Features

- Semantic document search
- TF-IDF + Truncated SVD embeddings
- Gaussian Mixture Model (GMM) clustering
- Semantic cache for faster repeated searches
- FastAPI REST API
- Interactive Swagger documentation

## Tech Stack

- Python
- FastAPI
- Scikit-learn
- NumPy
- NLTK
- Uvicorn

## Project Structure

```
SemanticSearchSystem/
│
├── api/
├── scripts/
├── dataset_raw/
├── cache_store/
├── vector_db/
├── data_loader.py
├── preprocessing.py
├── embedding_model.py
├── clustering.py
├── vector_store.py
├── semantic_cache.py
├── search_engine.py
└── README.md
```

## Workflow

Dataset → Preprocessing → Embedding → Clustering → Vector Store → Semantic Cache → Search API

## Status

✅ Project completed and tested locally.