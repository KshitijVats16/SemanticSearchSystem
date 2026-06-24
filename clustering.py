import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

DEFAULT_CLUSTER_DIR = "cache_store"
N_COMPONENTS_PCA = 50
COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42


class FuzzyClusterer:
    def __init__(
        self,
        n_clusters: int,
        pca_components: int = N_COMPONENTS_PCA,
        cluster_dir: str = DEFAULT_CLUSTER_DIR,
    ):
        self.n_clusters = n_clusters
        self.pca_components = pca_components

        self.cluster_dir = Path(cluster_dir)
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

        self._gmm_path = self.cluster_dir / "gmm.pkl"
        self._pca_path = self.cluster_dir / "pca.pkl"

        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None

    def fit(self, embeddings: np.ndarray):
        actual_components = min(
            self.pca_components,
            embeddings.shape[1],
            embeddings.shape[0] - 1,
        )

        print(f"Fitting PCA({actual_components})...")

        self.pca = PCA(
            n_components=actual_components,
            random_state=RANDOM_STATE,
        )

        reduced = self.pca.fit_transform(embeddings)

        print(
            f"PCA variance retained: {self.pca.explained_variance_ratio_.sum()*100:.2f}%"
        )

        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=COVARIANCE_TYPE,
            max_iter=200,
            random_state=RANDOM_STATE,
        )

        print("Training Gaussian Mixture Model...")
        self.gmm.fit(reduced)

        return self

    def predict_proba(self, embeddings: np.ndarray):
        reduced = self.pca.transform(embeddings)
        return self.gmm.predict_proba(reduced)

    def dominant_cluster(self, embeddings: np.ndarray):
        return self.predict_proba(embeddings).argmax(axis=1)

    def predict_single(
        self,
        embedding: np.ndarray,
    ) -> Tuple[int, np.ndarray]:

        proba = self.predict_proba(
            embedding.reshape(1, -1)
        )[0]

        return int(proba.argmax()), proba

    def save(self):
        with open(self._gmm_path, "wb") as f:
            pickle.dump(self.gmm, f)

        with open(self._pca_path, "wb") as f:
            pickle.dump(self.pca, f)

        print("Cluster model saved.")

    def load(self):
        if (
            not self._gmm_path.exists()
            or not self._pca_path.exists()
        ):
            return False

        with open(self._gmm_path, "rb") as f:
            self.gmm = pickle.load(f)

        with open(self._pca_path, "rb") as f:
            self.pca = pickle.load(f)

        self.n_clusters = self.gmm.n_components
        return True


def select_n_clusters(
    embeddings: np.ndarray,
    k_range=range(5, 26, 5),
    pca_components=N_COMPONENTS_PCA,
) -> Dict:

    actual = min(
        pca_components,
        embeddings.shape[1],
        embeddings.shape[0] - 1,
    )

    pca = PCA(
        n_components=actual,
        random_state=RANDOM_STATE,
    )

    reduced = pca.fit_transform(embeddings)

    results = {
        "k": [],
        "bic": [],
        "silhouette": [],
    }

    print(f"{'K':>5}{'BIC':>15}{'Silhouette':>15}")

    for k in k_range:

        gmm = GaussianMixture(
            n_components=k,
            covariance_type=COVARIANCE_TYPE,
            random_state=RANDOM_STATE,
        )

        gmm.fit(reduced)

        bic = gmm.bic(reduced)

        labels = gmm.predict(reduced)

        sil = silhouette_score(
            reduced,
            labels,
            sample_size=min(2000, len(reduced)),
        )

        results["k"].append(k)
        results["bic"].append(bic)
        results["silhouette"].append(sil)

        print(f"{k:>5}{bic:>15.2f}{sil:>15.4f}")

    return results