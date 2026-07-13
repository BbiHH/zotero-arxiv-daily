from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from dataclasses import dataclass
from loguru import logger
import requests

T = TypeVar("T")

HTML_EXTRACT_TIMEOUT = 180
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180
RETRIABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class NetworkPolicy:
    connect_timeout: float = 15
    read_timeout: float = 120
    max_attempts: int = 8
    retry_base_delay: float = 15
    retry_max_delay: float = 300

    @property
    def timeout(self) -> tuple[float, float]:
        return (self.connect_timeout, self.read_timeout)

    @classmethod
    def from_config(cls, config) -> "NetworkPolicy":
        config = config or {}
        return cls(
            connect_timeout=float(config.get("connect_timeout", cls.connect_timeout)),
            read_timeout=float(config.get("read_timeout", cls.read_timeout)),
            max_attempts=int(config.get("max_attempts", cls.max_attempts)),
            retry_base_delay=float(config.get("retry_base_delay", cls.retry_base_delay)),
            retry_max_delay=float(config.get("retry_max_delay", cls.retry_max_delay)),
        )


DEFAULT_NETWORK_POLICY = NetworkPolicy()


class _RetryableFeedError(RuntimeError):
    pass


class _TimeoutSession(requests.Session):
    """Requests session that supplies the timeout missing from arxiv.Client."""

    def __init__(self, policy: NetworkPolicy):
        super().__init__()
        self._timeout = policy.timeout

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self._timeout)
        return super().request(method, url, **kwargs)


def _status_code_from_error(exc: Exception) -> int | None:
    if isinstance(exc, arxiv.HTTPError):
        return exc.status
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_retryable(exc: Exception) -> bool:
    status_code = _status_code_from_error(exc)
    if status_code is not None:
        return status_code in RETRIABLE_STATUS_CODES
    return isinstance(
        exc,
        (
            requests.exceptions.RequestException,
            arxiv.UnexpectedEmptyPageError,
            _RetryableFeedError,
        ),
    )


def _retry_delay(policy: NetworkPolicy, attempt: int, exc: Exception) -> float:
    response = getattr(exc, "response", None)
    retry_after = getattr(response, "headers", {}).get("Retry-After") if response else None
    if retry_after:
        try:
            return min(float(retry_after), policy.retry_max_delay)
        except ValueError:
            pass
    return min(
        policy.retry_base_delay * (2 ** (attempt - 1)),
        policy.retry_max_delay,
    )


def _wait_to_retry(
    policy: NetworkPolicy,
    attempt: int,
    exc: Exception,
    operation: str,
) -> None:
    delay = _retry_delay(policy, attempt, exc)
    logger.warning(
        f"{operation} failed on attempt {attempt}/{policy.max_attempts}: {exc}; "
        f"retrying in {delay:g}s"
    )
    if delay > 0:
        sleep(delay)


def _request_content(url: str, policy: NetworkPolicy) -> bytes:
    for attempt in range(1, policy.max_attempts + 1):
        try:
            with requests.get(url, timeout=policy.timeout) as response:
                response.raise_for_status()
                return response.content
        except Exception as exc:
            if not _is_retryable(exc) or attempt == policy.max_attempts:
                raise
            _wait_to_retry(policy, attempt, exc, f"GET {url}")
    raise RuntimeError("unreachable")


def _fetch_rss_feed(url: str, policy: NetworkPolicy):
    for attempt in range(1, policy.max_attempts + 1):
        try:
            content = _request_content(url, NetworkPolicy(
                connect_timeout=policy.connect_timeout,
                read_timeout=policy.read_timeout,
                max_attempts=1,
                retry_base_delay=policy.retry_base_delay,
                retry_max_delay=policy.retry_max_delay,
            ))
            feed = feedparser.parse(content)
            if feed.bozo and not feed.entries:
                detail = feed.get("bozo_exception", "invalid RSS response")
                raise _RetryableFeedError(str(detail))
            return feed
        except Exception as exc:
            if not _is_retryable(exc) or attempt == policy.max_attempts:
                raise
            _wait_to_retry(policy, attempt, exc, "arXiv RSS request")
    raise RuntimeError("unreachable")


def _result_id(paper: ArxivResult) -> str:
    get_short_id = getattr(paper, "get_short_id", None)
    if callable(get_short_id):
        return get_short_id()
    return paper.entry_id.rstrip("/").rsplit("/", 1)[-1]


def _fetch_api_batch(
    client: arxiv.Client,
    paper_ids: list[str],
    policy: NetworkPolicy,
) -> list[ArxivResult]:
    results_by_id: dict[str, ArxivResult] = {}
    pending_ids = list(dict.fromkeys(paper_ids))

    for attempt in range(1, policy.max_attempts + 1):
        try:
            batch = list(client.results(arxiv.Search(id_list=pending_ids)))
            for paper in batch:
                paper_id = _result_id(paper)
                if paper_id in pending_ids:
                    results_by_id[paper_id] = paper
            pending_ids = [paper_id for paper_id in pending_ids if paper_id not in results_by_id]
            if not pending_ids:
                break
            exc = _RetryableFeedError(
                f"arXiv API returned a partial batch; {len(pending_ids)} IDs still missing"
            )
        except Exception as caught:
            if not _is_retryable(caught):
                raise
            exc = caught

        if attempt < policy.max_attempts:
            _wait_to_retry(policy, attempt, exc, "arXiv API batch")

    if pending_ids:
        logger.error(
            f"arXiv API retries exhausted; continuing without {len(pending_ids)} papers: "
            f"{', '.join(pending_ids)}"
        )
    return [results_by_id[paper_id] for paper_id in paper_ids if paper_id in results_by_id]


def _download_file(
    url: str,
    path: str,
    policy: NetworkPolicy | None = None,
) -> None:
    policy = policy or DEFAULT_NETWORK_POLICY
    for attempt in range(1, policy.max_attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=policy.timeout) as response:
                response.raise_for_status()
                with open(path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            return
        except Exception as exc:
            if not _is_retryable(exc) or attempt == policy.max_attempts:
                raise
            _wait_to_retry(policy, attempt, exc, f"Download {url}")


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str, policy: NetworkPolicy) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path, policy)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str, policy: NetworkPolicy) -> str | None:
    import trafilatura

    downloaded = _request_content(html_url, policy)
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(
    source_url: str,
    paper_id: str,
    paper_title: str | None,
    policy: NetworkPolicy,
) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path, policy)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")
        network_config = self.retriever_config.get("network") or {}
        extraction_config = self.retriever_config.get("extraction") or {}
        self.network_policy = NetworkPolicy.from_config(network_config)
        self.batch_size = int(network_config.get("batch_size", 20))
        self.api_delay_seconds = float(network_config.get("api_delay_seconds", 3))
        self.html_extract_timeout = float(
            extraction_config.get("html_timeout", HTML_EXTRACT_TIMEOUT)
        )
        self.pdf_extract_timeout = float(
            extraction_config.get("pdf_timeout", PDF_EXTRACT_TIMEOUT)
        )
        self.tar_extract_timeout = float(
            extraction_config.get("tar_timeout", TAR_EXTRACT_TIMEOUT)
        )

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(num_retries=0, delay_seconds=self.api_delay_seconds)
        if hasattr(client, "_session"):
            client._session = _TimeoutSession(self.network_policy)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = _fetch_rss_feed(
            f"https://rss.arxiv.org/atom/{query}", self.network_policy
        )
        if 'Feed error for query' in feed.feed.get("title", ""):
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0, len(all_paper_ids), self.batch_size):
            batch_ids = all_paper_ids[i:i + self.batch_size]
            batch = _fetch_api_batch(client, batch_ids, self.network_policy)
            bar.update(len(batch))
            raw_papers.extend(batch)
        bar.close()

        if len(raw_papers) != len(all_paper_ids):
            logger.warning(
                f"Retrieved metadata for {len(raw_papers)}/{len(all_paper_ids)} "
                "RSS paper IDs; continuing with the successful papers"
            )

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        published = getattr(raw_paper, "published", None)
        published_date = published.strftime("%Y-%m-%d") if published is not None else None
        source_url = raw_paper.source_url()
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            source_url=source_url,
            published_date=published_date,
            pdf_url=pdf_url,
            full_text=None,
        )

    def enrich_paper(self, paper: Paper) -> Paper:
        """Download full text only for papers that survive both ranking stages."""
        if paper.full_text is not None:
            return paper

        if paper.source_url:
            paper.full_text = _run_with_hard_timeout(
                _extract_text_from_tar_worker,
                (paper.source_url, paper.url, paper.title, self.network_policy),
                timeout=self.tar_extract_timeout,
                operation="Tar extraction",
                paper_title=paper.title,
            )
        if paper.full_text is None:
            html_url = paper.url.replace("/abs/", "/html/")
            paper.full_text = _run_with_hard_timeout(
                _extract_text_from_html_worker,
                (html_url, self.network_policy),
                timeout=self.html_extract_timeout,
                operation="HTML extraction",
                paper_title=paper.title,
            )
        if paper.full_text is None and paper.pdf_url:
            paper.full_text = _run_with_hard_timeout(
                _extract_text_from_pdf_worker,
                (paper.pdf_url, self.network_policy),
                timeout=self.pdf_extract_timeout,
                operation="PDF extraction",
                paper_title=paper.title,
            )
        return paper


def extract_text_from_html(
    paper: ArxivResult,
    policy: NetworkPolicy = DEFAULT_NETWORK_POLICY,
    timeout: float = HTML_EXTRACT_TIMEOUT,
) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    return _run_with_hard_timeout(
        _extract_text_from_html_worker,
        (html_url, policy),
        timeout=timeout,
        operation="HTML extraction",
        paper_title=paper.title,
    )


def extract_text_from_pdf(
    paper: ArxivResult,
    policy: NetworkPolicy = DEFAULT_NETWORK_POLICY,
    timeout: float = PDF_EXTRACT_TIMEOUT,
) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url, policy),
        timeout=timeout,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(
    paper: ArxivResult,
    policy: NetworkPolicy = DEFAULT_NETWORK_POLICY,
    timeout: float = TAR_EXTRACT_TIMEOUT,
) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title, policy),
        timeout=timeout,
        operation="Tar extraction",
        paper_title=paper.title,
    )
