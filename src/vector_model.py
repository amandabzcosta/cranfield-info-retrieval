from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class VectorSpaceModel:
    def __init__(self, doc_tokens: list[list[str]]):
        self.doc_tokens = doc_tokens
        doc_strings = [" ".join(tokens) for tokens in doc_tokens]

        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=lambda x: x.split(),
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False,
        )
        self.doc_matrix = self.vectorizer.fit_transform(doc_strings)
        self.vocab = self.vectorizer.get_feature_names_out()
        self.idf = self.vectorizer.idf_

    def rank(self, query_tokens: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        query_string = " ".join(query_tokens)
        query_vector = self.vectorizer.transform([query_string])

        scores = cosine_similarity(query_vector, self.doc_matrix)[0]

        doc_scores = list(enumerate(scores.tolist()))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        if top_n is not None:
            doc_scores = doc_scores[:top_n]
        return doc_scores
