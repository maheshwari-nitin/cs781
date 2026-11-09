"""PCR records, sentence text, split-specific pools and reproducible training pairs."""
import hashlib
import json
import random
import warnings


def case_id(value):
    if type(value) not in (str, int) or not str(value):
        raise ValueError(f"Expected a case ID string/integer, got {type(value).__name__}")
    return str(value)  # preserve existing leading zeros; never invent ID padding


def text_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if all(isinstance(s, str) for s in value):
            return "\n".join(value)
        if all(isinstance(s, dict) and isinstance(s.get("text"), str) for s in value):
            return "\n".join(s["text"] for s in value)
    raise ValueError("Case 'text' must be a string, list of sentences, or list of {text: string} records")


def load_cases(path, queries=False, labeled=True):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        rows = []
        for ident, value in raw.items():
            if isinstance(value, dict):
                if "id" in value and case_id(value["id"]) != case_id(ident):
                    raise ValueError(f"Mapping key disagrees with record ID: {ident}")
                rows.append(dict(value, id=ident))
            elif not queries:
                rows.append(dict(id=ident, text=value))
            else:
                raise ValueError("Query records must contain text and relevance fields")
    elif isinstance(raw, list):
        rows = raw
    else:
        raise ValueError("Expected list of records or ID-keyed object")
    out, seen = [], set()
    for record in rows:
        ident = case_id(record["id"])
        if ident in seen:
            raise ValueError(f"Duplicate case ID: {ident}")
        seen.add(ident)
        doc = dict(id=ident, text=text_value(record["text"]))
        if queries and labeled:
            refs = record.get("relevant_candidates")
            if not isinstance(refs, list):
                raise ValueError(f"{ident}: relevant_candidates must be a list")
            doc["relevant_candidates"] = sorted({case_id(v) for v in refs})
        # Unlabeled test queries and candidate documents never read relevance fields.
        out.append(doc)
    if not out:
        raise ValueError(f"Empty case file: {path}")
    return out


def load_split(root, split, labeled=True):
    from pathlib import Path
    root = Path(root)
    queries = load_cases(root / f"{split}.json", queries=True, labeled=labeled)
    pool = load_cases(root / f"{split}_candidates.json", labeled=False)
    valid = {d["id"] for d in pool}
    if labeled:
        for query in queries:
            missing = set(query["relevant_candidates"]) - valid
            if missing:
                raise ValueError(f"{split} query {query['id']}: relevance IDs outside its pool: {sorted(missing)[:5]}")
    return queries, pool


def fingerprint(queries, pool):
    h = hashlib.sha256()
    for name, docs in (("queries", queries), ("candidates", pool)):
        h.update(name.encode())
        for doc in sorted(docs, key=lambda d: d["id"]):
            h.update(json.dumps(doc, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            h.update(b"\n")
    return h.hexdigest()


def sample_pairs(queries, pool, negatives=4, max_positives=2, seed=1337,
                 hard_candidates=None, hard_fraction=0.5):
    """Seeded TRAIN negatives, excluding ALL positives; optionally mix mined hard and random IDs."""
    if not 0 <= hard_fraction <= 1:
        raise ValueError("hard_fraction must be in [0, 1]")
    ids = sorted(d["id"] for d in pool)
    rng = random.Random(seed)
    pairs, empty = [], []
    for query in sorted(queries, key=lambda d: d["id"]):
        gold = set(query["relevant_candidates"])
        if not gold:
            empty.append(query["id"])
            continue
        pos = sorted(gold)
        if max_positives and len(pos) > max_positives:
            pos = rng.sample(pos, max_positives)
        available = [i for i in ids if i not in gold]
        if not available:
            raise ValueError(f"Query {query['id']} has no negative candidates")
        take = min(negatives, len(available))
        hard = []
        if hard_candidates is not None:
            allowed = set(available)
            eligible = sorted(set(hard_candidates.get(query["id"], [])) & allowed)
            hard = rng.sample(eligible, min(round(take * hard_fraction), len(eligible)))
        chosen_hard = set(hard)
        remaining = [ident for ident in available if ident not in chosen_hard]
        neg = hard + rng.sample(remaining, take - len(hard))
        pairs.append(dict(query_id=query["id"], candidate_ids=pos + neg, n_positive=len(pos)))
        if hard_candidates is not None:
            pairs[-1]["n_hard_negative"] = len(hard)
    if empty:
        warnings.warn(f"Skipped {len(empty)} training queries without positive labels: {empty[:10]}", stacklevel=2)
    if not pairs:
        raise ValueError("No trainable queries with positive and negative candidates")
    return pairs, empty
