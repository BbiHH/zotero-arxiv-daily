from .base import BaseReranker, register_reranker
from .embedding_cache import EmbeddingCache, embedding_namespace
from openai import OpenAI
import numpy as np
@register_reranker("api")
class ApiReranker(BaseReranker):
    def get_similarity_score(
        self, candidate_documents: list[str], corpus_queries: list[str]
    ) -> np.ndarray:
        client = OpenAI(api_key=self.config.reranker.api.key, base_url=self.config.reranker.api.base_url)
        batch_size = self.config.reranker.api.get("batch_size") or 64
        model_name = str(self.config.reranker.api.model)
        endpoint = str(self.config.reranker.api.base_url)
        cache = EmbeddingCache(self.config)

        def encode(texts: list[str]) -> np.ndarray:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(input=batch, model=model_name)
                embeddings.extend([row.embedding for row in response.data])
            return np.asarray(embeddings)

        namespace = embedding_namespace(
            backend="api",
            model=model_name,
            role="symmetric",
            endpoint=endpoint,
        )
        document_embeddings = cache.get_or_compute(
            candidate_documents,
            namespace=namespace,
            label="API document/candidate",
            compute=encode,
        )
        query_embeddings = cache.get_or_compute(
            corpus_queries,
            namespace=namespace,
            label="API query/corpus",
            compute=encode,
        )
        document_embeddings = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        sim = np.dot(document_embeddings, query_embeddings.T)
        return sim
