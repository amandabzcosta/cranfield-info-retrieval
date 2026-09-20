import re
import sys
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DOCS_FILE = RAW_DIR / "cran.all.1400"
QUERIES_FILE = RAW_DIR / "cran.qry"
QRELS_FILE = RAW_DIR / "cranqrel"

# Marcadores de seção do formato SMART clássico do Cranfield
_SECTION_MARKERS = {".I", ".T", ".A", ".B", ".W"}

def _parse_smart_records(path: Path) -> list[dict]:
    """Lê um arquivo .I/.T/.A/.B/.W no formato SMART e retorna os registros."""
    records = []
    current = None
    current_section = None

    with open(path, "r", encoding="latin-1") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            marker = line[:2]

            if marker in _SECTION_MARKERS:
                if marker == ".I":
                    if current is not None:
                        records.append(current)
                    current = {".T": [], ".A": [], ".B": [], ".W": []}
                current_section = marker
                continue

            if current is not None and current_section in current:
                current[current_section].append(line)

        if current is not None:
            records.append(current)

    return records

def _load_docs_from_local(path: Path = DOCS_FILE) -> pd.DataFrame:
    records = _parse_smart_records(path)
    rows = []
    for doc_id, record in enumerate(records, start=1):
        rows.append(
            {
                "doc_id": str(doc_id),
                "title": "\n".join(record[".T"]).strip(),
                "text": "\n".join(record[".W"]).strip(),
                "author": "\n".join(record[".A"]).strip(),
                "bib": "\n".join(record[".B"]).strip(),
            }
        )
    return pd.DataFrame(rows)

def _load_queries_from_local(path: Path = QUERIES_FILE) -> pd.DataFrame:
    # Os números .I do cran.qry são históricos/não-sequenciais; o query_id
    # usado de fato pelo cranqrel (e pelo ir_datasets) é a posição da
    # consulta no arquivo (começando em 1), não o valor do .I.
    records = _parse_smart_records(path)
    rows = []
    for query_id, record in enumerate(records, start=1):
        rows.append(
            {
                "query_id": str(query_id),
                "text": "\n".join(record[".W"]).strip(),
            }
        )
    return pd.DataFrame(rows)

def _load_qrels_from_local(path: Path = QRELS_FILE) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            query_id, doc_id, relevance = parts[0], parts[1], parts[2]
            rows.append(
                {
                    "query_id": str(query_id),
                    "doc_id": str(doc_id),
                    "relevance": int(relevance),
                }
            )
    return pd.DataFrame(rows)

def _load_from_local_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        _load_docs_from_local(),
        _load_queries_from_local(),
        _load_qrels_from_local(),
    )

def load_cranfield() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        import ir_datasets

        dataset = ir_datasets.load("cranfield")
        df_docs = pd.DataFrame(dataset.docs_iter())
        df_queries = pd.DataFrame(dataset.queries_iter())
        df_qrels = pd.DataFrame(dataset.qrels_iter())
        return df_docs, df_queries, df_qrels
    except Exception as exc:
        print(
            f"[cranfield_data] ir_datasets indisponível ({type(exc).__name__}: {exc}). "
            "Usando arquivos locais em data/raw/ como fallback.",
            file=sys.stderr,
        )
        return _load_from_local_files()
