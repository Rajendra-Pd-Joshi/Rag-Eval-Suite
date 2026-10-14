# src/reranker.py
from sentence_transformers import CrossEncoder
from src.retriever import load_store
from langsmith import traceable

# small, fast, CPU-friendly reranker. Downloads once (~80MB) on first run.
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FETCH_K = 20   # how many the bi-encoder (nemotron-3-embed-1b) brings back
TOP_K = 5      # how many survive after reranking


class RerankingRetriever:
    def __init__(self, fetch_k=FETCH_K, top_k=TOP_K):
        self.store = load_store()
        self.reranker = CrossEncoder(CROSS_ENCODER)
        self.fetch_k = fetch_k   # over-retrieve
        self.top_k = top_k       # keep after reranking

    @traceable(run_type="retriever", name="RerankingRetriever")
    def invoke(self, query):
        # 1. OVER-RETRIEVE: fast bi-encoder, deliberately more than we need
        candidates = self.store.similarity_search(query, k=self.fetch_k)

        # 2. RERANK: score each (query, chunk) pair TOGETHER with the cross-encoder
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 3. SORT by score, keep the best top_k
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.top_k]]