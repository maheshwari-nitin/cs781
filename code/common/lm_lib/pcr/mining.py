"""Lexical hard-negative mining on the official TRAIN candidate pool only.

No pretrained encoder or external text. Reference API:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
"""


def mine_tfidf_negatives(queries, pool, top_k=64):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    if top_k < 1:
        raise ValueError("Hard-negative pool size must be positive")
    ordered = sorted(pool, key=lambda d: d["id"])
    ids = [d["id"] for d in ordered]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100_000,
                                sublinear_tf=True, dtype=np.float32,
                                token_pattern=r"(?u)\b\w+\b")
    try:
        candidates = vectorizer.fit_transform([d["text"] for d in ordered])
    except ValueError as error:
        if "empty vocabulary" not in str(error):
            raise
        return {q["id"]: [] for q in queries}  # blank corpus: sampler falls back to random
    result = {}
    for index, query in enumerate(queries, 1):
        q = vectorizer.transform([query["text"]])
        scores = (candidates @ q.T).toarray().ravel()
        gold = set(query["relevant_candidates"])
        order = np.argsort(-scores, kind="stable")
        result[query["id"]] = [ids[i] for i in order if scores[i] > 0 and ids[i] not in gold][:top_k]
        if index % 200 == 0 or index == len(queries):
            print(f"mined TRAIN lexical negatives {index}/{len(queries)}", flush=True)
    return result
