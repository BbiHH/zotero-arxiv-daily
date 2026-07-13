"""Tests for the persistent content-addressed embedding cache."""

import numpy as np
from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.embedding_cache import (
    EmbeddingCache,
    embedding_namespace,
)


def make_cache(tmp_path):
    config = OmegaConf.create({
        "reranker": {
            "embedding_cache": {
                "enabled": True,
                "directory": str(tmp_path),
            }
        }
    })
    return EmbeddingCache(config)


def test_cache_reuses_vectors_and_computes_only_unique_misses(tmp_path):
    cache = make_cache(tmp_path)
    namespace = embedding_namespace(
        backend="local", model="test/model", role="query", options={}
    )
    calls = []

    def compute(texts):
        calls.append(list(texts))
        return np.array([[len(text), len(text) + 1] for text in texts], dtype=np.float32)

    first = cache.get_or_compute(
        ["a", "bb", "a"], namespace=namespace, label="test", compute=compute
    )
    second = cache.get_or_compute(
        ["bb", "ccc"], namespace=namespace, label="test", compute=compute
    )

    assert calls == [["a", "bb"], ["ccc"]]
    np.testing.assert_array_equal(first, [[1, 2], [2, 3], [1, 2]])
    np.testing.assert_array_equal(second, [[2, 3], [3, 4]])


def test_cache_namespace_change_invalidates_vector(tmp_path):
    cache = make_cache(tmp_path)
    calls = []

    def compute(texts):
        calls.append(list(texts))
        return np.ones((len(texts), 2))

    first_namespace = embedding_namespace(
        backend="local", model="model-v1", role="document", options={}
    )
    second_namespace = embedding_namespace(
        backend="local", model="model-v2", role="document", options={}
    )
    cache.get_or_compute(["paper"], namespace=first_namespace, label="test", compute=compute)
    cache.get_or_compute(["paper"], namespace=second_namespace, label="test", compute=compute)

    assert calls == [["paper"], ["paper"]]
