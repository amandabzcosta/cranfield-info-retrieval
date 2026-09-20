from collections import defaultdict
from math import log

# Constrói o índice invertido usado pelo BM25
def build_bm25_index(doc_ids: list, docs: list[list[str]]) -> dict:
    N = len(docs)
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_lengths = []
    df = defaultdict(int)  # Document frequency de cada termo

    for doc_idx, doc_tokens in enumerate(docs):
        doc_length = len(doc_tokens)
        doc_lengths.append(doc_length)

        # Marca os termos únicos do documento, para calcular o df
        seen_terms = set()
        for term in doc_tokens:
            inverted_index[term][doc_idx] += 1
            seen_terms.add(term)

        for term in seen_terms:
            df[term] += 1

    # Comprimento médio dos documentos
    avg_doc_length = sum(doc_lengths) / N if N > 0 else 0

    return {
        "N": N,
        "doc_ids": doc_ids,
        "inverted_index": inverted_index,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "df": df,
    }

# Calcula o IDF de um termo pela fórmula do BM25
def bm25_idf(term: str, index: dict) -> float:
    N = index["N"]
    df_t = index["df"].get(term, 0)

    # Evita log de zero: termo que não aparece em nenhum doc tem IDF baixo
    if df_t == 0:
        return 0.0

    idf = log((N - df_t + 0.5) / (df_t + 0.5)) + 1
    return idf

# Calcula o score BM25 de um documento para uma consulta
def bm25_score(query_tokens: list[str], doc_idx: int, index: dict, k1: float = 1.2, b: float = 0.75) -> float:
    score = 0.0
    doc_length = index["doc_lengths"][doc_idx]
    avg_doc_length = index["avg_doc_length"]
    inverted_index = index["inverted_index"]

    for term in query_tokens:
        if term not in inverted_index:
            continue

        f_t_d = inverted_index[term][doc_idx]
        if f_t_d == 0:
            continue

        idf_t = bm25_idf(term, index)

        if idf_t == 0:
            continue

        # Numerador e denominador da fórmula do BM25
        numerator = f_t_d * (k1 + 1)
        denominator = f_t_d + k1 * (1 - b + b * doc_length / avg_doc_length)

        score += idf_t * (numerator / denominator)

    return score

# Rankeia documentos para cada consulta usando o score BM25
def rank_documents_bm25(
    doc_ids: list,
    query_ids: list,
    docs: list[list[str]],
    queries: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75
) -> dict:
    index = build_bm25_index(doc_ids, docs)

    rankings = {}
    for query_idx, query_id in enumerate(query_ids):
        query_tokens = queries[query_idx]

        # Calcula o score BM25 de cada documento
        doc_scores = []
        for doc_idx in range(len(doc_ids)):
            score = bm25_score(query_tokens, doc_idx, index, k1=k1, b=b)
            doc_scores.append((doc_ids[doc_idx], score))

        # Ordena por score decrescente
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        rankings[query_id] = doc_scores

    return rankings
