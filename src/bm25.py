from src.probabilistic_model import build_bm25_index, bm25_score

class BM25:
    """BM25 explícito (score calculado do zero, ver probabilistic_model.bm25_score).
    set_params troca k1/b sem reconstruir o índice invertido."""
    def __init__(self, doc_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.doc_tokens = doc_tokens
        self.k1 = k1
        self.b = b

        doc_ids = list(range(len(doc_tokens)))
        self.index = build_bm25_index(doc_ids, doc_tokens)

        self.n_docs = self.index["N"]
        self.avgdl = self.index["avg_doc_length"]
        self.inverted_index = self.index["inverted_index"]

    def set_params(self, k1: float = None, b: float = None) -> None:
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b

    def rank(self, query_tokens: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        doc_scores = [
            (doc_idx, bm25_score(query_tokens, doc_idx, self.index, k1=self.k1, b=self.b))
            for doc_idx in range(self.n_docs)
        ]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            doc_scores = doc_scores[:top_n]
        return doc_scores
