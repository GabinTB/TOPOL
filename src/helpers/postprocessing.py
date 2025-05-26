import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity as cosine

TOP_FREQ_N_WORDS = 20
TOP_REPR_DOCS = 10
TF_IDF_N_WORDS = 20
N_GRAMS_RANGE = (1, 2)
MAX_FEATURES = 10000

def get_top_n_words(documents: list, vectorizer_model: CountVectorizer, n: int=10) -> dict:
    words = vectorizer_model.fit_transform(documents)
    words_freq = words.sum(axis=0).A1
    words_freq = [(word, words_freq[idx]) for word, idx in vectorizer_model.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return dict(words_freq[:n])

def get_top_n_representative_documents(documents: list, embeddings: list, centroid: np.array, n: int=10) -> list:
    assert len(documents) == len(embeddings) > 0, "Documents and embeddings must have the same length."
    assert centroid.shape[0] == embeddings[0].shape[0], f"Centroid and embeddings must have the same dimensionality: {centroid.shape[0]} != {embeddings[0].shape[0]}"
    sim_matrix = cosine(embeddings, centroid.reshape(1, -1))
    sim_matrix = sim_matrix.ravel()
    sorted_indices = np.argsort(sim_matrix)[::-1]
    top_n_indices = sorted_indices[:min(len(sorted_indices), n)]
    return [documents[i] for i in top_n_indices]

def get_tf_idf_top_n_words(documents: list, vectorizer_model: CountVectorizer, n: int=10):
    doc_strings = documents
    X_counts = vectorizer_model.fit_transform(doc_strings)
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X_counts)
    feature_names = vectorizer_model.get_feature_names_out()
    sorted_indices = np.argsort(X_tfidf.toarray(), axis=1)[:, ::-1]
    top_n_words = []
    for i in range(X_tfidf.shape[0]):
        top_n_indices = sorted_indices[i, :n]
        top_n_words.append({feature_names[idx]: X_tfidf[i, idx] for idx in top_n_indices})
    return top_n_words