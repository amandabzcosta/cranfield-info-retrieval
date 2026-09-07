import math
import pandas as pd

DEFAULT_K = 10
ALL_METRICS = ["precision", "recall", "f1", "ap", "rr", "ndcg"]

# Build a dictionary mapping query_id to a set of relevant doc_ids
def build_relevant_sets(df_qrels: pd.DataFrame) -> dict:
    relevant = df_qrels[df_qrels["relevance"] >= 1]
    grouped = relevant.groupby("query_id")["doc_id"].apply(set)
    result = grouped.to_dict()

    for query_id in df_qrels["query_id"].unique():
        result.setdefault(query_id, set())

    return result

# Build a dictionary mapping query_id to a dictionary of doc_id to graded relevance
def build_graded_relevance(df_qrels: pd.DataFrame) -> dict:
    df = df_qrels.copy()
    # Cranfield's relevance scale is inverted: 1 = most relevant, 4 = least relevant,
    # -1 = not relevant. Flip it so higher values mean higher gain for NDCG.
    df["relevance_graded"] = df["relevance"].apply(lambda r: 5 - r if r >= 1 else 0)

    grouped = df.groupby("query_id").apply(
        lambda g: dict(zip(g["doc_id"], g["relevance_graded"])),
        include_groups=False,
    )
    return grouped.to_dict()

# Compute precision at k for a single query
def precision_at_k(retrieved: list, relevant: set, k: int = DEFAULT_K) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k

# Compute recall at k for a single query
def recall_at_k(retrieved: list, relevant: set, k: int = DEFAULT_K) -> float:
    if len(relevant) == 0:
        return float("nan")
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)

# Compute F1 score at k for a single query
def f1_at_k(retrieved: list, relevant: set, k: int = DEFAULT_K) -> float:
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)

    if math.isnan(recall):
        return float("nan")
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# Compute average precision (AP) for a single query
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

# Compute reciprocal rank (RR) for a single query
def reciprocal_rank(retrieved: list, relevant: set) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0

# Compute normalized discounted cumulative gain (NDCG) at k for a single query
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

# Evaluate the ranking performance across all queries
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

def compare_rankings(
    ranking_a: dict,
    ranking_b: dict,
    df_qrels: pd.DataFrame,
    k: int = DEFAULT_K,
    label_a: str = "model_a",
    label_b: str = "model_b",
    metrics: list = None,
) -> pd.DataFrame:
    per_query_a = evaluate_ranking(ranking_a, df_qrels, k, metrics)["per_query"]
    per_query_b = evaluate_ranking(ranking_b, df_qrels, k, metrics)["per_query"]

    merged = per_query_a.add_suffix(f"_{label_a}").join(
        per_query_b.add_suffix(f"_{label_b}"), how="outer"
    )

    for col in per_query_a.columns:
        if col in ("num_relevant", "num_retrieved"):
            continue
        merged[f"{col}_diff"] = merged[f"{col}_{label_b}"] - merged[f"{col}_{label_a}"]

    return merged

# Categorize queries based on the difference in performance metrics between two models
def categorize_query_differences(
    comparison_df: pd.DataFrame,
    metric: str = "ap",
    label_a: str = "model_a",
    label_b: str = "model_b",
    threshold: float = 0.3,
    bad_threshold: float = 0.1,
) -> dict:
    diff_col = f"{metric}_diff"
    col_a = f"{metric}_{label_a}"
    col_b = f"{metric}_{label_b}"

    categories = {
        "b_much_better_than_a": [],
        "a_much_better_than_b": [],
        "both_bad": [],
        "similar": [],
    }

    for query_id, row in comparison_df.iterrows():
        diff = row[diff_col]
        value_a = row[col_a]
        value_b = row[col_b]

        if pd.isna(diff):
            continue
        elif diff > threshold:
            categories["b_much_better_than_a"].append((query_id, diff))
        elif diff < -threshold:
            categories["a_much_better_than_b"].append((query_id, diff))
        elif value_a < bad_threshold and value_b < bad_threshold:
            categories["both_bad"].append((query_id, diff))
        else:
            categories["similar"].append((query_id, diff))

    categories["b_much_better_than_a"].sort(key=lambda x: x[1], reverse=True)
    categories["a_much_better_than_b"].sort(key=lambda x: x[1])
    categories["both_bad"].sort(key=lambda x: abs(x[1]))

    return {category: [query_id for query_id, _ in items] for category, items in categories.items()}

# Evaluate a grid of parameters for a ranking function and return a DataFrame with the results
def evaluate_parameter_grid(
    build_ranking_fn,
    param_grid: list,
    df_qrels: pd.DataFrame,
    k: int = DEFAULT_K,
    metrics: list = None,
) -> pd.DataFrame:
    rows = []
    for params in param_grid:
        ranking = build_ranking_fn(**params)
        result = evaluate_ranking(ranking, df_qrels, k, metrics)

        row = dict(params)
        row.update(result["aggregated"])
        rows.append(row)

    return pd.DataFrame(rows)
