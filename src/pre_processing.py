import nltk
import pandas as pd
import pickle
from pathlib import Path
from nltk.stem import PorterStemmer
from typing import Union


PREPROCESSING_CONFIGS = {
    "raw": {"remove_stopwords": False, "apply_stemming": False},
    "stopwords": {"remove_stopwords": True, "apply_stemming": False},
    "stemming": {"remove_stopwords": False, "apply_stemming": True},
    "stopwords_stemming": {"remove_stopwords": True, "apply_stemming": True},
}

STEMMER = PorterStemmer()
STOPWORDS = None


def ensure_nltk_resources():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    global STOPWORDS
    if STOPWORDS is None:
        STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def tokenize(text: str) -> list[str]:
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    filtered_tokens = [token for token in tokens if token.isalpha()]
    return filtered_tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    if STOPWORDS is None:
        ensure_nltk_resources()
    return [token for token in tokens if token not in STOPWORDS]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(token) for token in tokens]


def preprocess_text(
    text: str, remove_stopwords_flag: bool = False, apply_stemming_flag: bool = False
) -> list[str]:
    tokens = tokenize(text)

    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens)

    if apply_stemming_flag:
        tokens = stem_tokens(tokens)

    return tokens


def preprocess_corpus(
    texts: Union[pd.Series, list],
    remove_stopwords_flag: bool = False,
    apply_stemming_flag: bool = False,
) -> list[list[str]]:
    return [
        preprocess_text(text, remove_stopwords_flag, apply_stemming_flag) for text in texts
    ]


def get_document_text(df_docs: pd.DataFrame) -> pd.Series:
    return df_docs["title"] + " " + df_docs["text"]


def vocabulary_stats(tokenized_docs: list[list[str]]) -> dict:
    vocab = set()
    total_tokens = 0
    for tokens in tokenized_docs:
        vocab.update(tokens)
        total_tokens += len(tokens)

    avg_tokens_per_doc = total_tokens / len(tokenized_docs) if len(tokenized_docs) > 0 else 0

    return {
        "vocabulary_size": len(vocab),
        "total_tokens": total_tokens,
        "avg_tokens_per_doc": avg_tokens_per_doc,
    }


def build_all_configs(df_docs: pd.DataFrame, df_queries: pd.DataFrame) -> dict:
    ensure_nltk_resources()

    doc_texts = get_document_text(df_docs)
    query_texts = df_queries["text"]

    result = {}

    for config_name, config_flags in PREPROCESSING_CONFIGS.items():
        remove_sw = config_flags["remove_stopwords"]
        apply_stem = config_flags["apply_stemming"]

        preprocessed_docs = preprocess_corpus(doc_texts, remove_sw, apply_stem)
        preprocessed_queries = preprocess_corpus(query_texts, remove_sw, apply_stem)

        result[config_name] = {
            "doc_ids": df_docs["doc_id"].tolist(),
            "docs": preprocessed_docs,
            "query_ids": df_queries["query_id"].tolist(),
            "queries": preprocessed_queries,
        }

    return result


def save_preprocessed(data: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_preprocessed(path: Union[str, Path]) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)
