"""Micro metrics compare top seven against the COMPLETE gold set, without truncating gold."""


def validate_predictions(predictions, query_ids, candidate_ids, k=7):
    if set(predictions) != set(query_ids):
        raise ValueError("Predicted query IDs do not match the query split")
    valid = set(candidate_ids)
    if len(valid) < k:
        raise ValueError(f"Candidate pool contains fewer than {k} unique cases")
    for ident, ranking in predictions.items():
        if not isinstance(ranking, list) or len(ranking) != k or len(set(ranking)) != k:
            raise ValueError(f"{ident}: expected exactly {k} unique ranked candidate IDs")
        if any(not isinstance(v, str) or v not in valid for v in ranking):
            raise ValueError(f"{ident}: invalid candidate ID or candidate from another pool")


def retrieval_scores(predictions, queries, k=7):
    if set(predictions) != {d["id"] for d in queries}:
        raise ValueError("Metric query IDs do not match")
    tp = retrieved = relevant = 0
    per_query = []
    for query in queries:
        ranking = predictions[query["id"]][:k]
        if len(ranking) != len(set(ranking)):
            raise ValueError("Duplicate predictions cannot count as extra true positives")
        gold = set(query["relevant_candidates"])
        hits = len(set(ranking) & gold)
        tp += hits; retrieved += len(ranking); relevant += len(gold)
        per_query.append(dict(id=query["id"], hits=hits, relevant=len(gold), retrieved=len(ranking)))
    return dict(micro_f1_at7=2 * tp / (retrieved + relevant) if retrieved + relevant else 0.0,
                micro_precision_at7=tp / retrieved if retrieved else 0.0,
                micro_recall_at7=tp / relevant if relevant else 0.0,
                true_positives=tp, retrieved=retrieved, relevant=relevant, n_queries=len(queries),
                per_query=per_query)
