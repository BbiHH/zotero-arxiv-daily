"""Tests for ArxivRetriever."""

import time
from types import SimpleNamespace

import feedparser
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    NetworkPolicy,
    _download_file,
    _fetch_api_batch,
    _fetch_rss_feed,
    _run_with_hard_timeout,
    extract_text_from_html,
)
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        yield self.content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _result(paper_id: str):
    return SimpleNamespace(entry_id=f"https://arxiv.org/abs/{paper_id}")


def test_fetch_rss_retries_timeout_with_explicit_request_timeout(monkeypatch):
    calls = []
    sleeps = []
    rss = b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><title>ok</title></feed>'

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        if len(calls) == 1:
            raise requests.ReadTimeout("slow rss")
        return _FakeResponse(rss)

    monkeypatch.setattr(arxiv_retriever.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_retriever, "sleep", sleeps.append)
    policy = NetworkPolicy(
        connect_timeout=7,
        read_timeout=90,
        max_attempts=3,
        retry_base_delay=2,
        retry_max_delay=30,
    )

    feed = _fetch_rss_feed("https://rss.arxiv.org/atom/cs.AI", policy)

    assert feed.feed.title == "ok"
    assert len(calls) == 2
    assert calls[0][1]["timeout"] == (7, 90)
    assert sleeps == [2]


def test_fetch_api_batch_retries_only_missing_ids(monkeypatch):
    requested_batches = []

    class PartialClient:
        def results(self, search):
            requested_batches.append(search.id_list)
            if search.id_list == ["2601.00001v1", "2601.00002v1"]:
                return iter([_result("2601.00001v1")])
            return iter([_result("2601.00002v1")])

    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    papers = _fetch_api_batch(
        PartialClient(),
        ["2601.00001v1", "2601.00002v1"],
        NetworkPolicy(max_attempts=3, retry_base_delay=0),
    )

    assert [paper.entry_id.rsplit("/", 1)[-1] for paper in papers] == [
        "2601.00001v1",
        "2601.00002v1",
    ]
    assert requested_batches == [
        ["2601.00001v1", "2601.00002v1"],
        ["2601.00002v1"],
    ]


def test_fetch_api_batch_retries_network_timeout(monkeypatch):
    attempts = 0

    class TimeoutClient:
        def results(self, search):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise requests.ReadTimeout("slow api")
            return iter([_result(search.id_list[0])])

    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    papers = _fetch_api_batch(
        TimeoutClient(),
        ["2601.00001v1"],
        NetworkPolicy(max_attempts=3, retry_base_delay=0),
    )

    assert len(papers) == 1
    assert attempts == 2


def test_download_file_retries_transient_timeout(tmp_path, monkeypatch):
    attempts = 0

    def fake_get(url, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.ConnectTimeout("slow download")
        return _FakeResponse(b"paper bytes")

    monkeypatch.setattr(arxiv_retriever.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    output = tmp_path / "paper.pdf"

    _download_file(
        "https://arxiv.org/pdf/test",
        str(output),
        NetworkPolicy(max_attempts=3, retry_base_delay=0),
    )

    assert output.read_bytes() == b"paper bytes"
    assert attempts == 2


def test_download_file_retries_truncated_response(tmp_path, monkeypatch):
    attempts = 0

    def fake_get(url, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise requests.exceptions.ChunkedEncodingError("truncated response")
        return _FakeResponse(b"complete paper")

    monkeypatch.setattr(arxiv_retriever.requests, "get", fake_get)
    monkeypatch.setattr(arxiv_retriever, "sleep", lambda _: None)
    output = tmp_path / "paper.pdf"

    _download_file(
        "https://arxiv.org/pdf/test",
        str(output),
        NetworkPolicy(max_attempts=3, retry_base_delay=0),
    )

    assert output.read_bytes() == b"complete paper"
    assert attempts == 2


def test_html_extraction_uses_hard_timeout(monkeypatch):
    captured = {}

    def fake_run(func, args, *, timeout, operation, paper_title):
        captured.update(func=func, args=args, timeout=timeout, operation=operation)
        return "html text"

    monkeypatch.setattr(arxiv_retriever, "_run_with_hard_timeout", fake_run)
    paper = SimpleNamespace(
        entry_id="https://arxiv.org/abs/2601.00001",
        title="Paper",
    )

    result = extract_text_from_html(paper, timeout=240)

    assert result == "html text"
    assert captured["timeout"] == 240
    assert captured["operation"] == "HTML extraction"


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    monkeypatch.setattr(
        arxiv_retriever,
        "_fetch_rss_feed",
        lambda url, policy: mock_feedparser,
    )

    # The RSS fixture gives us paper IDs.  After feedparser, the code calls
    # arxiv.Client().results(search) which makes real HTTP requests.  We mock
    # the arxiv Client so the test stays offline.
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]
    paper_ids = [e.id.removeprefix("oai:arXiv.org:") for e in new_entries]

    # Build fake ArxivResult-like objects matching each RSS entry
    fake_results = []
    for entry in new_entries:
        pid = entry.id.removeprefix("oai:arXiv.org:")
        fake_results.append(SimpleNamespace(
            title=entry.title,
            authors=[SimpleNamespace(name="Test Author")],
            summary="Test abstract",
            pdf_url=f"https://arxiv.org/pdf/{pid}",
            entry_id=f"https://arxiv.org/abs/{pid}",
            source_url=lambda pid=pid: f"https://arxiv.org/e-print/{pid}",
        ))

    class FakeClient:
        def __init__(self, **kw):
            pass
        def results(self, search):
            return iter(fake_results)

    monkeypatch.setattr(arxiv_retriever.arxiv, "Client", FakeClient)

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda *a, **kw: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda *a, **kw: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda *a, **kw: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
