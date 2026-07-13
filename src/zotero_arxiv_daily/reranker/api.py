from .base import BaseReranker, register_reranker
from openai import OpenAI
import numpy as np
@register_reranker("api")
class ApiReranker(BaseReranker):
    def get_similarity_score(
        self, candidate_documents: list[str], corpus_queries: list[str]
    ) -> np.ndarray:
        client = OpenAI(api_key=self.config.reranker.api.key, base_url=self.config.reranker.api.base_url)
        batch_size = self.config.reranker.api.get("batch_size") or 64
        all_texts = candidate_documents + corpus_queries
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=self.config.reranker.api.model
            )
            all_embeddings.extend([r.embedding for r in response.data])
        document_embeddings = np.array(all_embeddings[:len(candidate_documents)])
        query_embeddings = np.array(all_embeddings[len(candidate_documents):])
        document_embeddings = document_embeddings / np.linalg.norm(
            document_embeddings, axis=1, keepdims=True
        )
        query_embeddings = query_embeddings / np.linalg.norm(
            query_embeddings, axis=1, keepdims=True
        )
        sim = np.dot(document_embeddings, query_embeddings.T)
        return sim
