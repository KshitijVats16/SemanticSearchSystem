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

    def fit(self, texts: List[str]) -> "EmbeddingModel":
        """Fit TF-IDF and SVD on the corpus."""
        print(f"[embedding_model] Fitting TF-IDF on {len(texts)} documents …")

        self.vectorizer = TfidfVectorizer(
            min_df=MIN_DF,
            max_features=MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
        )

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[embedding_model] TF-IDF shape: {tfidf_matrix.shape}")

        actual_dim = min(self.dim, tfidf_matrix.shape[1] - 1)
        print(f"[embedding_model] Fitting TruncatedSVD(n_components={actual_dim}) …")

        self.svd = TruncatedSVD(
            n_components=actual_dim,
            random_state=42,
        )

        self.svd.fit(tfidf_matrix)

        self.dim = actual_dim
        explained = self.svd.explained_variance_ratio_.sum()

        print(
            f"[embedding_model] SVD explains {explained*100:.1f}% of variance. dim={self.dim}"
        )

        return self