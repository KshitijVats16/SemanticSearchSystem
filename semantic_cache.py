"""
semantic_cache.py
-----------------
An in-memory semantic cache with cluster-scoped lookup.

Cache design rationale
-----------------------
Traditional caches key on exact string equality. A semantic cache keys on
meaning: two queries phrased differently but semantically equivalent should
share a cache entry.

Lookup algorithm
-----------------
1. Embed the incoming query → query_vec.
2. Ask the clusterer for its dominant cluster → cluster_id.
3. Only iterate over cache entries in that cluster bucket.
4. Compute cosine similarity between query_vec and each stored embedding.
5. If max_similarity >= threshold → cache hit.
6. Otherwise → cache miss.

Thread safety: NOT thread-safe.
Eviction: LRU-style per bucket.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logger_config import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD = 0.85
MAX_BUCKET_SIZE = 200


@dataclass
class CacheStats:
    hit_count: int = 0
    miss_count: int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "total_entries": self.total_entries,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(self.hit_rate, 4),
        }


class SemanticCache:
    """
    Cluster-partitioned semantic cache.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        max_bucket_size: int = MAX_BUCKET_SIZE,
    ):
        self.threshold = threshold
        self.max_bucket_size = max_bucket_size

        self._cache: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._stats = CacheStats()

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_id: int,
    ) -> Tuple[bool, Optional[str], Optional[str], float]:
        """
        Try to find a semantically similar cached query.
        """
        bucket = self._cache.get(cluster_id, [])

        best_score = -1.0
        best_entry = None

        for entry in bucket:
            score = float(np.dot(query_embedding, entry["embedding"]))

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry is not None:
            self._stats.hit_count += 1

            logger.info(
                f"Cache hit in cluster {cluster_id} "
                f"(similarity={best_score:.3f})"
            )

            return (
                True,
                best_entry["query"],
                best_entry["result"],
                best_score,
            )

        self._stats.miss_count += 1

        logger.info(
            f"Cache miss in cluster {cluster_id}"
        )

        return False, None, None, best_score

    def store(
        self,
        query: str,
        query_embedding: np.ndarray,
        cluster_id: int,
        result: str,
    ) -> None:
        """
        Insert a new entry into the appropriate cluster bucket.
        Evicts the oldest entry if the bucket exceeds max_bucket_size.
        """
        bucket = self._cache[cluster_id]

        if len(bucket) >= self.max_bucket_size:
            bucket.pop(0)

        bucket.append(
            {
                "query": query,
                "embedding": query_embedding.astype(np.float32),
                "result": result,
                "timestamp": time.time(),
            }
        )

        self._stats.total_entries = sum(
            len(b) for b in self._cache.values()
        )

        logger.info(
            f"Stored query in cluster {cluster_id}. "
            f"Bucket size={len(bucket)}"
        )

    def clear(self) -> None:
        """
        Remove all entries and reset statistics.
        """
        self._cache.clear()
        self._stats = CacheStats()

        logger.info("Semantic cache cleared")

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def bucket_sizes(self) -> Dict[int, int]:
        """
        Return a mapping of cluster_id → number of entries.
        """
        return {
            cid: len(bucket)
            for cid, bucket in self._cache.items()
        }