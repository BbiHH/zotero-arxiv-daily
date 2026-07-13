"""Tests for zotero_arxiv_daily.protocol: Paper.generate_tldr, Paper.generate_affiliations."""

import json

import pytest
from omegaconf import OmegaConf

from tests.canned_responses import make_sample_paper, make_stub_openai_client
from zotero_arxiv_daily.protocol import _task_generation_kwargs


@pytest.fixture()
def llm_params():
    return {
        "generation_kwargs": {"model": "gpt-4o-mini"},
        "tldr": {"max_attempts": 3, "retry_delay_seconds": 0},
        "affiliation": {"max_attempts": 3, "retry_delay_seconds": 0},
    }


# ---------------------------------------------------------------------------
# generate_tldr
# ---------------------------------------------------------------------------


def test_tldr_returns_response(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_tldr(client, llm_params)
    assert result == "Hello! How can I assist you today?"
    assert paper.tldr == result


def test_generation_kwargs_from_hydra_config_are_json_serializable(config):
    kwargs = _task_generation_kwargs(config.llm, "tldr")

    assert json.loads(json.dumps(kwargs)) == {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 350,
    }


@pytest.mark.parametrize("task", ["tldr", "affiliation"])
def test_generation_kwargs_recursively_convert_nested_hydra_values(task):
    llm_params = OmegaConf.create({
        "generation_kwargs": {
            "model": "gpt-4o-mini",
            "extra_body": {"provider": {"order": ["primary", "fallback"]}},
        },
        task: {"generation_kwargs": {"temperature": 0}},
    })

    kwargs = _task_generation_kwargs(llm_params, task)

    assert json.loads(json.dumps(kwargs)) == {
        "model": "gpt-4o-mini",
        "extra_body": {"provider": {"order": ["primary", "fallback"]}},
        "temperature": 0,
    }


def test_tldr_prompt_requests_only_high_level_experimental_effects(llm_params):
    captured = {}

    class Completions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            return type("Response", (), {
                "choices": [type("Choice", (), {
                    "message": type("Message", (), {"content": "中文总结"})()
                })()]
            })()

    client = type("Client", (), {
        "chat": type("Chat", (), {"completions": Completions()})()
    })()
    make_sample_paper().generate_tldr(client, llm_params)

    prompt = captured["messages"][1]["content"]
    assert "总体实验效果或结论" in prompt
    assert "无需追求具体指标、数值或完整实验细节" in prompt


def test_tldr_without_abstract_or_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(abstract="", full_text=None)
    result = paper.generate_tldr(client, llm_params)
    assert result == "TLDR 生成失败：本次无法生成可靠的中文总结，请查看英文摘要。"


def test_tldr_returns_chinese_failure_message_on_error(llm_params):
    paper = make_sample_paper()
    calls = 0

    # Client whose create() raises
    from types import SimpleNamespace

    def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("API down")

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=fail)
        )
    )
    result = paper.generate_tldr(broken_client, llm_params)
    assert result == "TLDR 生成失败：本次无法生成可靠的中文总结，请查看英文摘要。"
    assert calls == 3


def test_tldr_truncates_long_prompt(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text="word " * 10000)
    result = paper.generate_tldr(client, llm_params)
    assert result is not None


# ---------------------------------------------------------------------------
# generate_affiliations
# ---------------------------------------------------------------------------


def test_affiliations_returns_parsed_list(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert isinstance(result, list)
    assert "TsingHua University" in result
    assert "Peking University" in result


def test_affiliations_none_without_fulltext(llm_params):
    client = make_stub_openai_client()
    paper = make_sample_paper(full_text=None)
    result = paper.generate_affiliations(client, llm_params)
    assert result is None


def test_affiliations_deduplicates(llm_params):
    """Affiliation deduplication preserves the returned author order."""
    client = make_stub_openai_client()
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    assert len(result) == len(set(result))
    assert result == ["TsingHua University", "Peking University"]


def test_affiliations_malformed_llm_output(llm_params):
    """LLM returns affiliations without JSON brackets. Should fall back gracefully."""
    from types import SimpleNamespace

    calls = 0

    def create_no_brackets(**kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="TsingHua University, Peking University"),
                )
            ]
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_no_brackets)
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(client, llm_params)
    # Strict JSON parsing fails, so the public method returns None.
    assert result is None
    assert calls == 3


def test_affiliations_error_returns_none(llm_params):
    from types import SimpleNamespace

    broken_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        )
    )
    paper = make_sample_paper()
    result = paper.generate_affiliations(broken_client, llm_params)
    assert result is None
    assert paper.affiliations is None
