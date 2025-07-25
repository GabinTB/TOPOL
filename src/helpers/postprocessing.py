import numpy as np
import polars as pl
import json
import re
import ast
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity as cosine

TOP_FREQ_N_WORDS = 20
TOP_REPR_DOCS = 10
TF_IDF_N_WORDS = 20
N_GRAMS_RANGE = (1, 2)
MAX_FEATURES = 10000

def get_top_k_representative_docs(cb_cluster_info, cb_df, k=5):
    top_k_docs = []
    for row in cb_cluster_info.iter_rows(named=True):
        centroid = row["centroid"]
        cluster_id = row["cluster"]
        cluster_df = cb_df.filter(pl.col("cluster") == cluster_id)
        embeddings = cluster_df["embedding"].to_list()
        similarities = cosine([centroid], embeddings)[0]
        documents = cluster_df["text"].to_list()
        docs_sim_dict = { doc: sim for doc, sim in zip(documents, similarities) }
        sorted_docs = sorted(docs_sim_dict.items(), key=lambda x: x[1], reverse=True)
        top_k_docs_texts = [doc for doc, sim in sorted_docs[:k]]
        top_k_similarities = [sim for doc, sim in sorted_docs[:k]]
        top_k_docs.append({
            "cluster": cluster_id,
            "top_docs": top_k_docs_texts,
            "similarities": top_k_similarities
        })
    return pl.DataFrame(top_k_docs)

def get_top_n_words(documents: list, vectorizer_model: CountVectorizer, n: int=10) -> dict:
    words = vectorizer_model.fit_transform(documents)
    words_freq = words.sum(axis=0)
    words_freq = [(word, words_freq[idx]) for word, idx in vectorizer_model.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return dict(words_freq[:n])

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

def from_list_to_string(texts_list):
    # return '- \n\t'.join(texts_list)
    stringified_list = ""
    for i, text in enumerate(texts_list):
        if i == 0:
            stringified_list += f'\t- "{text}"'
        else:
            stringified_list += f'\n\t- "{text}"'
    return stringified_list

def safe_json_load(raw_response_text):
    # Strip leading/trailing whitespace and remove non-JSON "explanation" text if any
    raw_text = raw_response_text.strip()

    # Attempt quick fix: if it starts/ends with JSON brackets
    if not raw_text.startswith('[') and '[' in raw_text:
        raw_text = raw_text[raw_text.index('['):]
    if not raw_text.endswith(']') and ']' in raw_text:
        raw_text = raw_text[:raw_text.rindex(']') + 1]

    # Remove or escape invalid escape characters
    def escape_invalid_escapes(s):
        # Fix invalid escape sequences: \x, \u (if malformed), or backslashes not part of valid escape
        s = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)  # Replace lone backslashes
        return s

    try:
        return json.loads(escape_invalid_escapes(raw_text))
    except json.JSONDecodeError as e:
        try:
            # Try ast.literal_eval as fallback (tolerates single quotes, trailing commas)
            return ast.literal_eval(raw_text)
        except Exception as fallback_error:
            print("⚠️ JSON parsing failed.")
            print("JSON error:", e)
            print("Fallback error:", fallback_error)
            return None
