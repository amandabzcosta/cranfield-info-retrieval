import math
import pandas as pd

DEFAULT_K = 10
ALL_METRICS = ["precision", "recall", "f1", "ap", "rr", "ndcg"]

# Mapeia cada query_id ao conjunto de doc_ids relevantes
def build_relevant_sets(df_qrels: pd.DataFrame) -> dict:
    relevant = df_qrels[df_qrels["relevance"] >= 1]
    grouped = relevant.groupby("query_id")["doc_id"].apply(set)
    result = grouped.to_dict()

    for query_id in df_qrels["query_id"].unique():
        result.setdefault(query_id, set())

    return result

# Mapeia cada query_id a um dicionário doc_id -> relevância graduada
def build_graded_relevance(df_qrels: pd.DataFrame) -> dict:
    df = df_qrels.copy()
    # A escala de relevância do Cranfield é invertida: 1 = mais relevante, 4 = menos
    # relevante, -1 = não relevante. Inverte para que valores maiores = mais ganho no NDCG.
    df["relevance_graded"] = df["relevance"].apply(lambda r: 5 - r if r >= 1 else 0)

    grouped = df.groupby("query_id").apply(
        lambda g: dict(zip(g["doc_id"], g["relevance_graded"])),
        include_groups=False,
    )
    return grouped.to_dict()

# Calcula precision@k para uma consulta
def precision_at_k(retrieved: list, relevant: set, k: int = DEFAULT_K) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k

# Calcula recall@k para uma consulta
def recall_at_k(retrieved: list, relevant: set, k: int = DEFAULT_K) -> float:
    if len(relevant) == 0:
        return float("nan")
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)

# Calcula F1@k para uma consulta
def f1_at_k(retrieved: list, relevant: set, k: int = DEFAULT_K) -> float:
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    if math.isnan(recall):
        return float("nan")
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Calcula average precision (AP) para uma consulta
def average_precision(retrieved: list, relevant: set) -> float:
    if len(relevant) == 0:
        return float("nan")

    hits = 0
    sum_precisions = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            sum_precisions += hits / i

    return sum_precisions / len(relevant)

# Calcula reciprocal rank (RR) para uma consulta
def reciprocal_rank(retrieved: list, relevant: set) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

# Calcula NDCG@k para uma consulta
def ndcg_at_k(retrieved: list, graded_relevance: dict, k: int = DEFAULT_K) -> float:
    def dcg(doc_ids_ordered):
        total = 0.0
        for i, doc_id in enumerate(doc_ids_ordered[:k], start=1):
            rel = graded_relevance.get(doc_id, 0)
            total += (2 ** rel - 1) / math.log2(i + 1)
        return total

    dcg_at_k = dcg(retrieved)
    ideal_order = sorted(graded_relevance.keys(), key=lambda d: graded_relevance[d], reverse=True)
    idcg_at_k = dcg(ideal_order)

    if idcg_at_k == 0:
        return float("nan")
    return dcg_at_k / idcg_at_k

# Avalia o desempenho do ranking em todas as consultas
def evaluate_ranking(
    ranking: dict,
    df_qrels: pd.DataFrame,
    k: int = DEFAULT_K,
    metrics: list = None,
) -> dict:
    metrics = metrics or ALL_METRICS
    relevant_sets = build_relevant_sets(df_qrels)
    graded_relevance = build_graded_relevance(df_qrels) if "ndcg" in metrics else {}

    rows = {}
    for query_id, doc_score_list in ranking.items():
        retrieved = [doc_id for doc_id, _ in doc_score_list]
        relevant = relevant_sets.get(query_id, set())

        row = {"num_relevant": len(relevant), "num_retrieved": len(retrieved)}
        if "precision" in metrics:
            row[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        if "recall" in metrics:
            row[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        if "f1" in metrics:
            row[f"f1@{k}"] = f1_at_k(retrieved, relevant, k)
        if "ap" in metrics:
            row["ap"] = average_precision(retrieved, relevant)
        if "rr" in metrics:
            row["rr"] = reciprocal_rank(retrieved, relevant)
        if "ndcg" in metrics:
            graded_q = graded_relevance.get(query_id, {})
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved, graded_q, k)

        rows[query_id] = row

    per_query = pd.DataFrame.from_dict(rows, orient="index")
    per_query.index.name = "query_id"

    aggregated = {}
    for col in per_query.columns:
        if col in ("num_relevant", "num_retrieved"):
            continue
        aggregated[col] = per_query[col].mean(skipna=True)

    if "ap" in aggregated:
        aggregated["map"] = aggregated.pop("ap")
    if "rr" in aggregated:
        aggregated["mrr"] = aggregated.pop("rr")

    aggregated["num_queries"] = len(per_query)
    aggregated["num_queries_with_relevant"] = int((per_query["num_relevant"] > 0).sum())

    return {"per_query": per_query, "aggregated": aggregated}
