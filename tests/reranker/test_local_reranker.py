"""Tests for LocalReranker query/document encoding."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.local import LocalReranker


def test_default_local_config_uses_qwen_and_scientific_query_instruction(config):
    local_config = config.reranker.local

    assert local_config.model == "Qwen/Qwen3-Embedding-0.6B"
    assert "scientific paper" in local_config.query_encode_kwargs.prompt
    assert local_config.query_encode_kwargs.normalize_embeddings is True
    assert local_config.document_encode_kwargs.normalize_embeddings is True
    assert "prompt" not in local_config.document_encode_kwargs


def test_local_reranker_encodes_corpus_as_query_and_candidates_as_documents(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, trust_remote_code):
            assert model == "Qwen/Qwen3-Embedding-0.6B"
            assert trust_remote_code is True

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.ones((len(texts), 2))

        def similarity(self, documents, queries):
            return np.ones((len(documents), len(queries)))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    config = OmegaConf.create(
        {
            "executor": {"debug": True},
            "reranker": {
                "local": {
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "query_encode_kwargs": {"prompt": "scientific query: "},
                    "document_encode_kwargs": {"normalize_embeddings": True},
                }
            },
        }
    )

    score = LocalReranker(config).get_similarity_score(
        ["candidate document"], ["zotero interest query"]
    )

    assert score.shape == (1, 1)
    assert calls[0][0] == ["zotero interest query"]
    assert calls[0][1]["prompt"] == "scientific query: "
    assert calls[1][0] == ["candidate document"]
    assert calls[1][1]["normalize_embeddings"] is True


def test_local_reranker_legacy_encode_kwargs_apply_to_both_sides(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, trust_remote_code):
            pass

        def encode(self, texts, **kwargs):
            calls.append(kwargs)
            return np.ones((len(texts), 2))

        def similarity(self, documents, queries):
            return np.ones((len(documents), len(queries)))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    config = OmegaConf.create(
        {
            "executor": {"debug": True},
            "reranker": {
                "local": {
                    "model": "legacy/model",
                    "encode_kwargs": {"normalize_embeddings": True},
                }
            },
        }
    )

    LocalReranker(config).get_similarity_score(["candidate"], ["corpus"])

    assert calls[0]["normalize_embeddings"] is True
    assert calls[1]["normalize_embeddings"] is True


def test_local_reranker_reuses_cached_embeddings(monkeypatch, tmp_path):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, trust_remote_code):
            pass

        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.ones((len(texts), 2))

        def similarity(self, documents, queries):
            return np.ones((len(documents), len(queries)))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    config = OmegaConf.create({
        "executor": {"debug": True},
        "reranker": {
            "embedding_cache": {
                "enabled": True,
                "directory": str(tmp_path),
            },
            "local": {
                "model": "test/model",
                "query_encode_kwargs": {"prompt": "query: "},
                "document_encode_kwargs": {"normalize_embeddings": True},
            },
        },
    })

    LocalReranker(config).get_similarity_score(["candidate"], ["corpus"])
    first_call_count = len(calls)
    LocalReranker(config).get_similarity_score(["candidate"], ["corpus"])

    assert first_call_count == 2
    assert len(calls) == first_call_count


@pytest.mark.slow
def test_local_reranker(config):
    reranker = LocalReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)
