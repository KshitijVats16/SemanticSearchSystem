class EmbeddingModel:
    """
    TF-IDF + Truncated SVD (LSA) embedding model.
    """

    def __init__(self, dim: int = DEFAULT_DIM, model_dir: str = MODEL_DIR):
        self.dim = dim
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model_name = f"lsa-tfidf-d{dim}"

        self._tfidf_path = self.model_dir / "tfidf.pkl"
        self._svd_path = self.model_dir / "svd.pkl"

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.svd: Optional[TruncatedSVD] = None