from abc import ABC, abstractmethod
from numbers import Real
from omegaconf import DictConfig, ListConfig
from ..protocol import Paper, CorpusPaper
from ..utils import glob_match
import numpy as np
from typing import Type


def normalize_collection_priority(
    priorities: list[dict] | ListConfig | None,
) -> list[tuple[str, float]]:
    """Validate collection priority rules while preserving configuration order."""
    if priorities is None:
        return []
    if not isinstance(priorities, (list, ListConfig)):
        raise TypeError("config.zotero.collection_priority must be a list or null")

    normalized = []
    for priority in priorities:
        if not hasattr(priority, "keys") or set(priority.keys()) != {"pattern", "weight"}:
            raise TypeError(
                "each config.zotero.collection_priority item must contain only pattern and weight"
            )
        pattern = priority["pattern"]
        weight = priority["weight"]
        if not isinstance(pattern, str) or not pattern:
            raise TypeError("collection priority pattern must be a non-empty string")
        if isinstance(weight, bool) or not isinstance(weight, Real) or weight <= 0:
            raise ValueError("collection priority weight must be a positive number")
        normalized.append((pattern, float(weight)))
    return normalized


class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        candidate_documents = [
            f"Title: {candidate.title}\nAbstract: {candidate.abstract}"
            for candidate in candidates
        ]
        corpus_queries = [
            f"Title: {paper.title}\nAbstract: {paper.abstract}"
            for paper in corpus
        ]
        sim = self.get_similarity_score(candidate_documents, corpus_queries)
        assert sim.shape == (len(candidates), len(corpus))

        configured_priorities = None
        if self.config is not None and self.config.get("zotero") is not None:
            configured_priorities = self.config.zotero.get("collection_priority")
        priorities = normalize_collection_priority(configured_priorities)

        if not priorities:
            scores = sim.mean(axis=1) * 10
        else:
            group_members: dict[int, list[int]] = {}
            for corpus_index, paper in enumerate(corpus):
                matched_rules = [
                    (rule_index, weight)
                    for rule_index, (pattern, weight) in enumerate(priorities)
                    if any(glob_match(path, pattern) for path in paper.paths)
                ]
                if not matched_rules:
                    continue
                # A paper contributes to one group only. Higher weight wins;
                # configuration order breaks equal-weight ties deterministically.
                rule_index, _ = max(matched_rules, key=lambda item: (item[1], -item[0]))
                group_members.setdefault(rule_index, []).append(corpus_index)

            if not group_members:
                raise ValueError(
                    "config.zotero.collection_priority patterns matched no papers "
                    "after include_path/ignore_path filtering"
                )

            active_weights = np.array(
                [priorities[index][1] for index in group_members], dtype=float
            )
            group_means = np.stack(
                [
                    sim[:, member_indexes].mean(axis=1)
                    for member_indexes in group_members.values()
                ],
                axis=1,
            )
            scores = np.average(group_means, axis=1, weights=active_weights) * 10

        for s,c in zip(scores,candidates):
            c.embedding_score = float(s)
            c.score = float(s)
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(
        self, candidate_documents: list[str], corpus_queries: list[str]
    ) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
