"""Tests for ApiReranker — uses stub OpenAI client via monkeypatch."""

from zotero_arxiv_daily.reranker.api import ApiReranker


def test_api_reranker_similarity_shape(config, patch_openai):
    reranker = ApiReranker(config)
    score = reranker.get_similarity_score(["hello", "world"], ["ping"])
    assert score.shape == (2, 1)


def test_api_reranker_batching(config, patch_openai):
    reranker = ApiReranker(config)
    s1 = [f"text {i}" for i in range(5)]
    s2 = [f"corpus {i}" for i in range(3)]
    score = reranker.get_similarity_score(s1, s2)
    assert score.shape == (5, 3)


def test_api_reranker_reuses_cached_embeddings(config, patch_openai, tmp_path):
    calls = []
    original_create = patch_openai.embeddings.create

    def tracked_create(**kwargs):
        calls.append(list(kwargs["input"]))
        return original_create(**kwargs)

    patch_openai.embeddings.create = tracked_create
    config.reranker.embedding_cache.enabled = True
    config.reranker.embedding_cache.directory = str(tmp_path)

    ApiReranker(config).get_similarity_score(["candidate"], ["corpus"])
    first_call_count = len(calls)
    ApiReranker(config).get_similarity_score(["candidate"], ["corpus"])

    assert first_call_count == 2
    assert len(calls) == first_call_count
