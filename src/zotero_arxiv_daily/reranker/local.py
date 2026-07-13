from .base import BaseReranker, register_reranker
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
        shared_kwargs = dict(local_config.get("encode_kwargs") or {})
        query_kwargs = shared_kwargs | dict(local_config.get("query_encode_kwargs") or {})
        document_kwargs = shared_kwargs | dict(local_config.get("document_encode_kwargs") or {})
        query_kwargs.setdefault("show_progress_bar", True)
        document_kwargs.setdefault("show_progress_bar", True)

        query_features = encoder.encode(corpus_queries, **query_kwargs)
        document_features = encoder.encode(candidate_documents, **document_kwargs)
        sim = encoder.similarity(document_features, query_features)
        if hasattr(sim, "detach"):
            return sim.detach().cpu().numpy()
        return np.asarray(sim)
