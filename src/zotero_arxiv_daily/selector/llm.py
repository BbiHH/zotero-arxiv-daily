import json
from numbers import Real
from pathlib import Path
from time import sleep

from loguru import logger
from omegaconf import DictConfig, OmegaConf
from openai import OpenAI

from ..protocol import Paper


class InvalidScoreResponse(ValueError):
    """Raised when an LLM response cannot be safely matched to the input papers."""


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidScoreResponse(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_score_response(
    content: str,
    expected_ids: set[str],
    *,
    score_step: int,
) -> dict[str, int]:
    """Parse and strictly validate an ID-to-score response."""
    try:
        payload = json.loads(content, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InvalidScoreResponse(f"response is not strict JSON: {exc}") from exc

    if not isinstance(payload, dict) or set(payload) != {"scores"}:
        raise InvalidScoreResponse("response must contain only the top-level 'scores' object")
    scores = payload["scores"]
    if not isinstance(scores, dict):
        raise InvalidScoreResponse("'scores' must be an object keyed by paper ID")

    returned_ids = set(scores)
    if returned_ids != expected_ids:
        missing = sorted(expected_ids - returned_ids)
        unknown = sorted(returned_ids - expected_ids)
        raise InvalidScoreResponse(
            f"paper ID mismatch; missing={missing}, unknown={unknown}"
        )

    for paper_id, score in scores.items():
        if isinstance(score, bool) or not isinstance(score, int):
            raise InvalidScoreResponse(f"score for {paper_id} must be an integer")
        if not 0 <= score <= 100:
            raise InvalidScoreResponse(f"score for {paper_id} must be between 0 and 100")
        if score % score_step != 0:
            raise InvalidScoreResponse(
                f"score for {paper_id} must be a multiple of {score_step}"
            )
    return scores


def embedding_percentile_scores(count: int) -> list[float]:
    """Map an already embedding-sorted candidate list onto a stable 0-100 scale."""
    if count <= 0:
        return []
    if count == 1:
        return [100.0]
    return [100.0 * (count - rank - 1) / (count - 1) for rank in range(count)]


class LLMPaperSelector:
    """Select a small email set from embedding candidates using robust LLM scoring."""

    def __init__(self, config: DictConfig, client: OpenAI):
        self.llm_config = config.llm
        self.config = config.llm.filter
        self.client = client

        self.batch_size = self._positive_int("batch_size")
        self.output_paper_num = self._positive_int("output_paper_num")
        self.max_candidates = int(config.executor.max_paper_num)
        if self.max_candidates <= 0:
            raise ValueError("executor.max_paper_num must be a positive integer")
        self.max_attempts = self._positive_int("max_attempts")
        self.score_step = self._positive_int("score_step")
        if 100 % self.score_step != 0:
            raise ValueError("llm.filter.score_step must divide 100 exactly")

        self.retry_delay_seconds = float(self.config.get("retry_delay_seconds") or 0)
        if self.retry_delay_seconds < 0:
            raise ValueError("llm.filter.retry_delay_seconds must be non-negative")

        weights = self.config.weights
        self.llm_weight = self._weight(weights.get("llm"), "llm")
        self.embedding_weight = self._weight(weights.get("embedding"), "embedding")
        total_weight = self.llm_weight + self.embedding_weight
        if total_weight <= 0:
            raise ValueError("llm.filter.weights must contain at least one positive weight")
        self.llm_weight /= total_weight
        self.embedding_weight /= total_weight

        self.prompt_template = self._load_prompt(self.config.prompt_file)

    def _positive_int(self, key: str) -> int:
        value = self.config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"llm.filter.{key} must be a positive integer")
        return value

    @staticmethod
    def _weight(value, key: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real) or value < 0:
            raise ValueError(f"llm.filter.weights.{key} must be a non-negative number")
        return float(value)

    @staticmethod
    def _load_prompt(prompt_file: str) -> str:
        path = Path(prompt_file).expanduser()
        candidates = [path] if path.is_absolute() else [
            Path.cwd() / path,
            Path(__file__).resolve().parents[3] / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8")
        raise FileNotFoundError(f"LLM filter prompt file not found: {prompt_file}")

    def _render_prompt(self, batch: list[tuple[str, Paper]], correction: str = "") -> str:
        papers = [
            {"id": paper_id, "title": paper.title, "abstract": paper.abstract}
            for paper_id, paper in batch
        ]
        replacements = {
            "{{research_profile}}": str(self.config.get("research_profile") or ""),
            "{{screening_requirements}}": str(
                self.config.get("screening_requirements") or ""
            ),
            "{{score_step}}": str(self.score_step),
            "{{papers_json}}": json.dumps(papers, ensure_ascii=False, indent=2),
            "{{correction}}": correction,
        }
        prompt = self.prompt_template
        for marker, value in replacements.items():
            prompt = prompt.replace(marker, value)
        return prompt

    def _generation_kwargs(self) -> dict:
        base = self.llm_config.get("generation_kwargs") or {}
        overrides = self.config.get("generation_kwargs") or {}
        if isinstance(base, DictConfig):
            base = OmegaConf.to_container(base, resolve=True)
        if isinstance(overrides, DictConfig):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        kwargs = dict(base) | dict(overrides)
        if self.config.get("use_json_mode", False):
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _score_batch(self, batch: list[tuple[str, Paper]]) -> dict[str, int]:
        expected_ids = {paper_id for paper_id, _ in batch}
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            correction = ""
            if last_error is not None:
                correction = (
                    "Your previous response was invalid. Correct the format and return the "
                    f"complete ID-to-score object. Validation error: {last_error}"
                )
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a conservative scientific-paper screening model. "
                                "Follow the requested JSON contract exactly."
                            ),
                        },
                        {"role": "user", "content": self._render_prompt(batch, correction)},
                    ],
                    **self._generation_kwargs(),
                )
                content = response.choices[0].message.content
                return parse_score_response(
                    content,
                    expected_ids,
                    score_step=self.score_step,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    f"LLM paper scoring attempt {attempt}/{self.max_attempts} failed: {exc}"
                )
                if attempt < self.max_attempts and self.retry_delay_seconds > 0:
                    sleep(self.retry_delay_seconds)
        raise InvalidScoreResponse(f"LLM scoring failed after retries: {last_error}")

    def select(self, candidates: list[Paper]) -> list[Paper]:
        if not candidates:
            return []

        working = list(candidates[:self.max_candidates])
        if len(candidates) > self.max_candidates:
            logger.warning(
                f"LLM filter can score at most {self.max_candidates} papers; "
                f"ignoring {len(candidates) - self.max_candidates} lower-ranked candidates"
            )

        percentile_scores = embedding_percentile_scores(len(working))
        paper_ids = [f"paper_{index + 1:04d}" for index in range(len(working))]
        for index, paper in enumerate(working):
            paper.embedding_rank = index + 1
            if paper.embedding_score is None:
                paper.embedding_score = float(paper.score or 0.0)

        fallback_count = 0
        for start in range(0, len(working), self.batch_size):
            batch_papers = working[start:start + self.batch_size]
            batch_ids = paper_ids[start:start + self.batch_size]
            batch = list(zip(batch_ids, batch_papers))
            try:
                scores = self._score_batch(batch)
                fallback = False
            except Exception as exc:
                logger.error(
                    f"Falling back to embedding scores for batch "
                    f"{start // self.batch_size + 1}: {exc}"
                )
                # Use the exact embedding percentile for both weighted terms so
                # the final score cleanly reduces to embedding-only ranking.
                scores = {
                    paper_id: percentile_scores[start + offset]
                    for offset, paper_id in enumerate(batch_ids)
                }
                fallback = True
                fallback_count += len(batch_papers)

            for offset, (paper_id, paper) in enumerate(batch):
                embedding_score = percentile_scores[start + offset]
                paper.llm_score = float(scores[paper_id])
                paper.final_score = (
                    self.llm_weight * paper.llm_score
                    + self.embedding_weight * embedding_score
                )
                paper.score = paper.final_score
                paper.selection_fallback = fallback

        working.sort(
            key=lambda paper: (
                -(paper.final_score or 0.0),
                -(paper.llm_score or 0.0),
                -(paper.embedding_score or 0.0),
                paper.embedding_rank or 10**9,
                paper.url,
            )
        )
        selected = working[: min(self.output_paper_num, len(working))]
        logger.info(
            f"LLM filter selected {len(selected)}/{len(working)} papers; "
            f"embedding fallback used for {fallback_count} papers"
        )
        return selected
