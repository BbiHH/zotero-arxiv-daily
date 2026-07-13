"""Persistent, content-addressed cache for embedding vectors."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
from loguru import logger
from omegaconf import OmegaConf


CACHE_SCHEMA_VERSION = 1


def plain_mapping(value: Mapping | None) -> dict:
    """Return a recursively plain mapping suitable for calls and fingerprints."""
    value = value or {}
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    return dict(value)


def embedding_namespace(
    *,
    backend: str,
    model: str,
    role: str,
    options: Mapping | None = None,
    endpoint: str | None = None,
) -> str:
    """Fingerprint every setting that can change the resulting vectors."""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "backend": backend,
        "model": model,
        "role": role,
        "options": plain_mapping(options),
        "endpoint": endpoint,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class EmbeddingCache:
    """SQLite-backed vector cache that fails open when storage is unavailable."""

    def __init__(self, config):
        cache_config = config.reranker.get("embedding_cache")
        self.enabled = bool(cache_config and cache_config.get("enabled", False))
        self.database_path: Path | None = None

        if not self.enabled:
            return

        cache_dir = Path(str(cache_config.get("directory"))).expanduser()
        self.database_path = cache_dir / f"embeddings-v{CACHE_SCHEMA_VERSION}.sqlite3"
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        cache_key TEXT PRIMARY KEY,
                        dtype TEXT NOT NULL,
                        dimension INTEGER NOT NULL,
                        vector BLOB NOT NULL
                    )
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            logger.warning(f"Embedding cache unavailable; continuing without it: {exc}")
            self.enabled = False
            self.database_path = None

    def _connect(self) -> sqlite3.Connection:
        if self.database_path is None:
            raise RuntimeError("embedding cache is disabled")
        return sqlite3.connect(self.database_path, timeout=30)

    @staticmethod
    def _key(namespace: str, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(namespace.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _validate_computed(vectors, expected_count: int) -> np.ndarray:
        array = np.asarray(vectors)
        if array.ndim != 2 or array.shape[0] != expected_count:
            raise ValueError(
                "embedding backend returned an invalid shape: "
                f"expected ({expected_count}, dimension), got {array.shape}"
            )
        return array

    def get_or_compute(
        self,
        texts: Sequence[str],
        *,
        namespace: str,
        label: str,
        compute: Callable[[list[str]], np.ndarray],
    ) -> np.ndarray:
        """Return vectors in input order, computing only unique cache misses."""
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=float)
        if not self.enabled:
            return self._validate_computed(compute(texts), len(texts))

        keys = [self._key(namespace, text) for text in texts]
        unique_keys = list(dict.fromkeys(keys))
        cached: dict[str, np.ndarray] = {}

        try:
            with self._connect() as connection:
                for start in range(0, len(unique_keys), 900):
                    chunk = unique_keys[start:start + 900]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = connection.execute(
                        f"SELECT cache_key, dtype, dimension, vector "
                        f"FROM embeddings WHERE cache_key IN ({placeholders})",
                        chunk,
                    )
                    for cache_key, dtype, dimension, blob in rows:
                        try:
                            vector = np.frombuffer(blob, dtype=np.dtype(dtype)).copy()
                            if vector.ndim == 1 and vector.size == dimension:
                                cached[cache_key] = vector
                        except (TypeError, ValueError):
                            continue
        except (OSError, sqlite3.Error) as exc:
            logger.warning(f"Embedding cache read failed for {label}; recomputing: {exc}")
            return self._validate_computed(compute(texts), len(texts))

        missing_keys = [key for key in unique_keys if key not in cached]
        if missing_keys:
            first_text_by_key = {}
            for key, text in zip(keys, texts):
                first_text_by_key.setdefault(key, text)
            missing_texts = [first_text_by_key[key] for key in missing_keys]
            computed = self._validate_computed(compute(missing_texts), len(missing_texts))
            for key, vector in zip(missing_keys, computed):
                cached[key] = np.asarray(vector)

            try:
                with self._connect() as connection:
                    connection.executemany(
                        """
                        INSERT OR REPLACE INTO embeddings
                            (cache_key, dtype, dimension, vector)
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (
                                key,
                                cached[key].dtype.str,
                                int(cached[key].size),
                                sqlite3.Binary(cached[key].tobytes()),
                            )
                            for key in missing_keys
                        ],
                    )
            except (OSError, sqlite3.Error) as exc:
                logger.warning(f"Embedding cache write failed for {label}: {exc}")

        hit_count = len(texts) - sum(key in missing_keys for key in keys)
        logger.info(
            f"Embedding cache {label}: {hit_count}/{len(texts)} hits; "
            f"computed {len(missing_keys)} unique misses"
        )
        return np.stack([cached[key] for key in keys])
