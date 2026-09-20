import pandas as pd
from src.evaluation import evaluate_ranking

_COLUMN_RENAME = {
    "ap": "AP",
    "rr": "MRR",
    "num_relevant": "n_relevant",
}

def evaluate_model(rankings: dict, df_qrels: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    # evaluate_ranking espera pares (doc_id, score), mas só usa a ordem —
    # então basta dar um score decrescente qualquer pra cada doc_id.
    scored_rankings = {
        query_id: [(doc_id, len(doc_ids) - pos) for pos, doc_id in enumerate(doc_ids)]
        for query_id, doc_ids in rankings.items()
    }

    result = evaluate_ranking(scored_rankings, df_qrels, k=k)
    per_query = result["per_query"].rename(columns=_COLUMN_RENAME)
    per_query = per_query.rename(columns={
        f"f1@{k}": f"F1@{k}",
    })
    per_query.index = per_query.index.astype(str)
    per_query.index.name = "query_id"

    column_order = [f"precision@{k}", f"recall@{k}", "AP", f"F1@{k}", "MRR", f"ndcg@{k}", "n_relevant"]
    column_order = [c for c in column_order if c in per_query.columns]
    return per_query[column_order]
