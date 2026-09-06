from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import dict, list, tuple


def build_tfidf_matrices(docs: list[list[str]], queries: list[list[str]]) -> tuple:
    """
    Build TF-IDF matrices for documents and queries.

    Args:
        docs: List of tokenized documents (list of token lists)
        queries: List of tokenized queries (list of token lists)

    Returns:
        Tuple of (doc_matrix, query_matrix, vectorizer)
    """
    # Convert token lists back to strings for TfidfVectorizer
    doc_strings = [" ".join(tokens) for tokens in docs]
    query_strings = [" ".join(tokens) for tokens in queries]

    # Initialize TfidfVectorizer with custom tokenizer to prevent re-tokenization
    # Use analyzer='word' to treat pre-tokenized strings as single tokens per space
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=lambda x: x.split(),  # Split by space (already tokenized)
        preprocessor=lambda x: x,  # No preprocessing (already lowercased and processed)
        token_pattern=None,  # Disable default token pattern
        lowercase=False  # Already lowercased in preprocessing
    )

    # Fit on documents and transform both documents and queries
    doc_matrix = vectorizer.fit_transform(doc_strings)
    query_matrix = vectorizer.transform(query_strings)

    return doc_matrix, query_matrix, vectorizer


def rank_documents_vectorial(
    doc_ids: list,
    query_ids: list,
    docs: list[list[str]],
    queries: list[list[str]]
) -> dict:
    """
    Rank documents using Vector Space Model (TF-IDF + Cosine Similarity).

    Args:
        doc_ids: List of document IDs (aligned with docs)
        query_ids: List of query IDs (aligned with queries)
        docs: List of tokenized documents
        queries: List of tokenized queries

    Returns:
        Dictionary mapping query_id -> list of (doc_id, score) tuples, sorted by score descending
    """
    doc_matrix, query_matrix, _ = build_tfidf_matrices(docs, queries)

    # Compute cosine similarity between queries and documents
    similarity_matrix = cosine_similarity(query_matrix, doc_matrix)

    rankings = {}
    for query_idx, query_id in enumerate(query_ids):
        scores = similarity_matrix[query_idx]

        # Create list of (doc_id, score) tuples
        doc_scores = [(doc_ids[doc_idx], float(scores[doc_idx]))
                      for doc_idx in range(len(doc_ids))]

        # Sort by score descending
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        rankings[query_id] = doc_scores

    return rankings
