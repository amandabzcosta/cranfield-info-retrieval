from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constrói as matrizes TF-IDF de documentos e consultas
def build_tfidf_matrices(docs: list[list[str]], queries: list[list[str]]) -> tuple:
    # Volta as listas de tokens para strings, para o TfidfVectorizer
    doc_strings = [" ".join(tokens) for tokens in docs]
    query_strings = [" ".join(tokens) for tokens in queries]

    # Tokenizer customizado para não retokenizar o que já foi pré-processado
    # analyzer='word' trata cada string pré-tokenizada como tokens separados por espaço
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=lambda x: x.split(),  # Separa por espaço (já tokenizado)
        preprocessor=lambda x: x,  # Sem pré-processamento (já feito antes)
        token_pattern=None,  # Desliga o padrão de token default
        lowercase=False  # Já veio em minúsculas do pré-processamento
    )

    # Ajusta nos documentos e transforma documentos e consultas
    doc_matrix = vectorizer.fit_transform(doc_strings)
    query_matrix = vectorizer.transform(query_strings)

    return doc_matrix, query_matrix, vectorizer

# Rankeia documentos para cada consulta usando o Modelo Vetorial (TF-IDF + cosseno)
def rank_documents_vectorial(
    doc_ids: list,
    query_ids: list,
    docs: list[list[str]],
    queries: list[list[str]]) -> dict:

    doc_matrix, query_matrix, _ = build_tfidf_matrices(docs, queries)

    # Similaridade de cosseno entre consultas e documentos
    similarity_matrix = cosine_similarity(query_matrix, doc_matrix)

    rankings = {}
    for query_idx, query_id in enumerate(query_ids):
        scores = similarity_matrix[query_idx]

        # Lista de tuplas (doc_id, score)
        doc_scores = [(doc_ids[doc_idx], float(scores[doc_idx]))
                      for doc_idx in range(len(doc_ids))]

        # Ordena por score decrescente
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        rankings[query_id] = doc_scores

    return rankings
