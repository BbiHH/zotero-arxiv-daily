"""Tests for BaseReranker collection-aware scoring and registration."""

from datetime import datetime

import numpy as np
import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.reranker.base import (
    BaseReranker,
    get_reranker_cls,
    normalize_collection_priority,
)
from zotero_arxiv_daily.protocol import CorpusPaper
from tests.canned_responses import make_sample_paper, make_sample_corpus


class StubReranker(BaseReranker):
    """Reranker with a controlled similarity matrix for deterministic tests."""

    def __init__(self, sim_matrix: np.ndarray, collection_priority=None):
        self.config = OmegaConf.create(
            {"zotero": {"collection_priority": collection_priority}}
        )
        self._sim = sim_matrix

    def get_similarity_score(self, s1, s2):
        self.seen_documents = s1
        self.seen_queries = s2
        return self._sim


def test_rerank_scores_and_sorts():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title=f"Paper {i}") for i in range(2)]

    # Paper 1 has higher similarity to all corpus papers
    sim = np.array([
        [0.1, 0.1, 0.1],  # paper 0 — low
        [0.9, 0.9, 0.9],  # paper 1 — high
    ])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].title == "Paper 1"
    assert ranked[1].title == "Paper 0"
    assert ranked[0].score > ranked[1].score
    assert reranker.seen_documents[0].startswith("Title: Paper 0\nAbstract:")
    assert reranker.seen_queries[0].startswith("Title: Corpus Paper 0\nAbstract:")


def test_rerank_without_collection_priority_uses_plain_corpus_mean():
    corpus = make_sample_corpus(3)
    papers = [make_sample_paper(title="P")]
    sim = np.array([[0.0, 0.0, 0.9]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert ranked[0].score == pytest.approx(3.0)


def test_rerank_collection_priority_averages_within_collection_then_weights_groups():
    corpus = [
        CorpusPaper("High A", "a", datetime(2024, 1, 1), ["library/high/a"]),
        CorpusPaper("High B", "b", datetime(2026, 1, 1), ["library/high/b"]),
        CorpusPaper("Low", "c", datetime(2025, 1, 1), ["library/low/c"]),
    ]
    papers = [make_sample_paper(title="Candidate")]
    priorities = [
        {"pattern": "library/high/**", "weight": 3},
        {"pattern": "library/low/**", "weight": 1},
    ]
    reranker = StubReranker(np.array([[1.0, 0.0, 1.0]]), priorities)

    ranked = reranker.rerank(papers, corpus)

    # High collection mean = 0.5, low collection mean = 1.0.
    # Final normalized score = (0.5 * 3 + 1.0 * 1) / 4 * 10.
    assert ranked[0].score == pytest.approx(6.25)


def test_rerank_overlap_assigns_paper_only_to_highest_weight_pattern():
    corpus = [
        CorpusPaper(
            "Overlap",
            "a",
            datetime(2026, 1, 1),
            ["library/low/a", "library/high/a"],
        ),
        CorpusPaper("Low only", "b", datetime(2026, 1, 2), ["library/low/b"]),
    ]
    priorities = [
        {"pattern": "library/low/**", "weight": 1},
        {"pattern": "library/high/**", "weight": 3},
    ]
    reranker = StubReranker(np.array([[1.0, 0.0]]), priorities)

    ranked = reranker.rerank([make_sample_paper()], corpus)

    # The overlap paper contributes once, to high. Low contains only the zero-sim paper.
    assert ranked[0].score == pytest.approx(7.5)


def test_rerank_collection_priority_rejects_when_no_corpus_paper_matches():
    corpus = [
        CorpusPaper("Other", "a", datetime(2026, 1, 1), ["library/other/a"])
    ]
    priorities = [{"pattern": "library/high/**", "weight": 3}]
    reranker = StubReranker(np.array([[1.0]]), priorities)

    with pytest.raises(ValueError, match="collection_priority patterns matched no papers"):
        reranker.rerank([make_sample_paper()], corpus)


def test_collection_priority_schema_allows_only_pattern_and_weight():
    with pytest.raises(TypeError, match="contain only pattern and weight"):
        normalize_collection_priority(
            [{"pattern": "library/high/**", "weight": 3, "name": "high"}]
        )


@pytest.mark.parametrize("weight", [0, -1, True])
def test_collection_priority_weight_must_be_positive_number(weight):
    with pytest.raises(ValueError, match="positive number"):
        normalize_collection_priority(
            [{"pattern": "library/high/**", "weight": weight}]
        )


def test_rerank_single_candidate_single_corpus():
    corpus = make_sample_corpus(1)
    papers = [make_sample_paper()]
    sim = np.array([[0.5]])
    reranker = StubReranker(sim)
    ranked = reranker.rerank(papers, corpus)
    assert len(ranked) == 1
    assert ranked[0].score is not None


def test_get_reranker_cls_unknown():
    with pytest.raises(ValueError, match="not found"):
        get_reranker_cls("nonexistent_reranker_xyz")
