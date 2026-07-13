from .base import BaseReranker, register_reranker
from .embedding_cache import EmbeddingCache, embedding_namespace, plain_mapping
import logging
import warnings
import numpy as np
@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(
        self, candidate_documents: list[str], corpus_queries: list[str]
    ) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        encoder = SentenceTransformer(self.config.reranker.local.model, trust_remote_code=True)
        local_config = self.config.reranker.local
        shared_kwargs = plain_mapping(local_config.get("encode_kwargs"))
        query_kwargs = shared_kwargs | plain_mapping(local_config.get("query_encode_kwargs"))
        document_kwargs = shared_kwargs | plain_mapping(
            local_config.get("document_encode_kwargs")
        )
        query_kwargs.setdefault("show_progress_bar", True)
        document_kwargs.setdefault("show_progress_bar", True)

        cache = EmbeddingCache(self.config)
        model_name = str(local_config.model)
        query_features = cache.get_or_compute(
            corpus_queries,
            namespace=embedding_namespace(
                backend="local",
                model=model_name,
                role="query",
                options=query_kwargs,
            ),
            label="local query/corpus",
            compute=lambda texts: encoder.encode(texts, **query_kwargs),
        )
        document_features = cache.get_or_compute(
            candidate_documents,
            namespace=embedding_namespace(
                backend="local",
                model=model_name,
                role="document",
                options=document_kwargs,
            ),
            label="local document/candidate",
            compute=lambda texts: encoder.encode(texts, **document_kwargs),
        )
        sim = encoder.similarity(document_features, query_features)
        if hasattr(sim, "detach"):
            return sim.detach().cpu().numpy()
        return np.asarray(sim)
