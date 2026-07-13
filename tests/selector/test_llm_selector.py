import json
import re
from types import SimpleNamespace

import pytest
from omegaconf import open_dict

from tests.canned_responses import make_sample_paper
from zotero_arxiv_daily.selector.llm import (
    InvalidScoreResponse,
    LLMPaperSelector,
    parse_score_response,
)


def _response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class QueueClient:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _response(self.contents.pop(0))


class EchoClient:
    def __init__(self, score=50):
        self.score = score
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        )

    def create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][1]["content"]
        paper_ids = re.findall(r'"id": "(paper_\d+)"', prompt)
        return _response(json.dumps({"scores": {paper_id: self.score for paper_id in paper_ids}}))


def _selector_config(config, **overrides):
    with open_dict(config):
        config.llm.filter.enabled = True
        config.llm.filter.batch_size = overrides.get("batch_size", 5)
        config.llm.filter.max_batches = overrides.get("max_batches", 20)
        config.llm.filter.output_paper_num = overrides.get("output_paper_num", 30)
        config.llm.filter.max_attempts = overrides.get("max_attempts", 3)
        config.llm.filter.retry_delay_seconds = 0
        config.llm.filter.score_step = 5
        config.llm.filter.research_profile = "visual reasoning profile"
        config.llm.filter.screening_requirements = "prefer concrete methods"
    return config


def _papers(count):
    papers = []
    for index in range(count):
        paper = make_sample_paper(
            title=f"Paper {index}",
            url=f"https://arxiv.org/abs/2607.{index:05d}",
            score=10.0 - index,
            embedding_score=10.0 - index,
        )
        papers.append(paper)
    return papers


def test_parse_score_response_rejects_duplicate_ids():
    content = '{"scores":{"paper_0001":80,"paper_0001":60}}'

    with pytest.raises(InvalidScoreResponse, match="duplicate JSON key"):
        parse_score_response(content, {"paper_0001"}, score_step=5)


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"scores":{"paper_9999":80}}', "paper ID mismatch"),
        ('{"scores":{"paper_0001":81}}', "multiple of 5"),
        ('{"scores":{"paper_0001":"80"}}', "must be an integer"),
        ('```json\n{"scores":{"paper_0001":80}}\n```', "not strict JSON"),
    ],
)
def test_parse_score_response_rejects_unsafe_outputs(content, message):
    with pytest.raises(InvalidScoreResponse, match=message):
        parse_score_response(content, {"paper_0001"}, score_step=5)


def test_selector_matches_scores_by_id_and_combines_embedding(config):
    config = _selector_config(config, output_paper_num=2)
    client = QueueClient(
        ['{"scores":{"paper_0003":0,"paper_0001":50,"paper_0002":100}}']
    )

    selected = LLMPaperSelector(config, client).select(_papers(3))

    assert [paper.title for paper in selected] == ["Paper 1", "Paper 0"]
    assert selected[0].llm_score == 100
    assert selected[0].final_score == pytest.approx(85.0)
    assert selected[1].final_score == pytest.approx(65.0)
    prompt = client.calls[0]["messages"][1]["content"]
    assert "visual reasoning profile" in prompt
    assert "prefer concrete methods" in prompt
    assert "embedding_score" not in prompt
    assert client.calls[0]["model"] == config.llm.generation_kwargs.model
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["max_tokens"] == 800


def test_selector_scores_five_papers_per_batch(config):
    config = _selector_config(config, batch_size=5, max_batches=20)
    client = EchoClient(score=50)

    selected = LLMPaperSelector(config, client).select(_papers(12))

    assert len(client.calls) == 3
    assert len(selected) == 12
    assert all(paper.llm_score == 50 for paper in selected)


def test_selector_retries_id_mismatch_then_accepts_complete_response(config):
    config = _selector_config(config, output_paper_num=2, max_attempts=2)
    client = QueueClient(
        [
            '{"scores":{"paper_0001":80}}',
            '{"scores":{"paper_0001":80,"paper_0002":60}}',
        ]
    )

    selected = LLMPaperSelector(config, client).select(_papers(2))

    assert len(client.calls) == 2
    assert all(not paper.selection_fallback for paper in selected)
    assert "previous response was invalid" in client.calls[1]["messages"][1]["content"].lower()


def test_selector_falls_back_to_embedding_for_failed_batch(config):
    config = _selector_config(config, output_paper_num=3, max_attempts=2)
    client = QueueClient(["not json", "still not json"])

    selected = LLMPaperSelector(config, client).select(_papers(3))

    assert [paper.title for paper in selected] == ["Paper 0", "Paper 1", "Paper 2"]
    assert [paper.final_score for paper in selected] == pytest.approx([100, 50, 0])
    assert all(paper.selection_fallback for paper in selected)


def test_selector_caps_work_to_batch_size_times_max_batches(config):
    config = _selector_config(
        config,
        batch_size=2,
        max_batches=2,
        output_paper_num=10,
    )
    client = EchoClient(score=50)

    selected = LLMPaperSelector(config, client).select(_papers(6))

    assert len(client.calls) == 2
    assert len(selected) == 4
