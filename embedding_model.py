import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from logger_config import get_logger

logger = get_logger(__name__)

DEFAULT_DIM = 256
MIN_DF = 2
MAX_FEATURES = 50000
MODEL_DIR = "cache_store"


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
        logger.info(f"Fitting TF-IDF on {len(texts)} documents")

        self.vectorizer = TfidfVectorizer(
            min_df=MIN_DF,
            max_features=MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
        )

        tfidf_matrix = self.vectorizer.fit_transform(texts)

        logger.info(f"TF-IDF shape: {tfidf_matrix.shape}")

        actual_dim = min(self.dim, tfidf_matrix.shape[1] - 1)

        self.svd = TruncatedSVD(
            n_components=actual_dim,
            random_state=42,
        )

        self.svd.fit(tfidf_matrix)

        self.dim = actual_dim

        explained = self.svd.explained_variance_ratio_.sum()

        logger.info(
            f"SVD explains {explained * 100:.1f}% of variance. dim={self.dim}"
        )

        return self

    def save(self):
        with open(self._tfidf_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(self._svd_path, "wb") as f:
            pickle.dump(self.svd, f)

        logger.info(f"Saved TF-IDF + SVD to {self.model_dir}")

    def load(self):
        if not self._tfidf_path.exists() or not self._svd_path.exists():
            return False

        with open(self._tfidf_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(self._svd_path, "rb") as f:
            self.svd = pickle.load(f)

        self.dim = self.svd.n_components

        logger.info(
            f"Loaded TF-IDF + SVD (dim={self.dim}) from {self.model_dir}"
        )

        return True

    def embed(
        self,
        texts: List[str],
        batch_size: int = 512,
        show_progress: bool = True,
    ) -> np.ndarray:
        assert self.vectorizer is not None and self.svd is not None, (
            "Call fit() or load() before embed()."
        )

        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        vecs = normalize(vecs, norm="l2")

        return vecs.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text], show_progress=False)[0]


_model_instance: Optional[EmbeddingModel] = None


def get_model(
    dim: int = DEFAULT_DIM,
    model_dir: str = MODEL_DIR,
) -> EmbeddingModel:
    global _model_instance

    if _model_instance is None:
        _model_instance = EmbeddingModel(
            dim=dim,
            model_dir=model_dir,
        )

        if not _model_instance.load():
            raise RuntimeError(
                f"Embedding model not found in '{model_dir}'. "
                "Run scripts/build_index.py first."
            )

    return _model_instance


if __name__ == "__main__":
    texts = [
        "Astronomy and space exploration missions",
        "GPU drivers crash on Windows 10",
        "Space telescope discovers new exoplanet",
    ]

    model = EmbeddingModel()
    model.fit(texts)

    vecs = model.embed(texts)

    print(f"Shape : {vecs.shape}")
    print(f"Norm  : {np.linalg.norm(vecs[0]):.4f}")
    print(f"sim(0,2) = {np.dot(vecs[0], vecs[2]):.4f}")
    print(f"sim(0,1) = {np.dot(vecs[0], vecs[1]):.4f}")