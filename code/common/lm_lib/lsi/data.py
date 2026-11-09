"""Validated LSI records and a persisted, non-contiguous integer statute vocabulary."""
import json

from ..cjpe.data import prepare as prepare_chunks


def load_split(path, labeled=True):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, (list, dict)):
        raise ValueError("Expected a list of records or an ID-to-record object")
    items = raw if isinstance(raw, list) else [dict(v, id=k) for k, v in raw.items()]
    docs, seen = [], set()
    for x in items:
        ident = str(x["id"])
        if ident in seen:
            raise ValueError(f"Duplicate document ID: {ident}")
        seen.add(ident)
        if not isinstance(x["text"], str):
            raise ValueError(f"Non-string text for {ident}")
        d = dict(id=ident, text=x["text"])
        if labeled:
            labels = x.get("labels")
            if not isinstance(labels, list) or any(type(v) is not int for v in labels):
                raise ValueError(f"{ident}: labels must be a list of integer statute IDs")
            # Released annotations can repeat a statute. Multi-label targets are sets:
            # count each statute once for targets, frequencies and loss weights.
            d["labels"] = sorted(set(labels))
        # In inference mode the labels field is never accessed, including if present.
        docs.append(d)
    if not docs:
        raise ValueError(f"Empty split: {path}")
    return docs


def label_vocab(train, dev, explicit=None):
    """Use released train/dev label IDs as metadata, never test labels or guessed ranges."""
    observed = {v for d in train + dev for v in d["labels"]}
    labels = sorted(observed if explicit is None else explicit)
    if not labels or any(type(v) is not int for v in labels) or len(labels) != len(set(labels)):
        raise ValueError("Statute vocabulary must contain unique integer IDs")
    if observed - set(labels):
        raise ValueError(f"Vocabulary misses released labels: {sorted(observed - set(labels))}")
    return labels


def check_disjoint(**splits):
    seen = {}
    for split, docs in splits.items():
        for d in docs:
            if d["id"] in seen:
                raise ValueError(f"ID {d['id']} overlaps {seen[d['id']]} and {split}")
            seen[d["id"]] = split


def prepare(docs, tok, cfg):
    # Reuse the Phase-4 tokenization/chunking code; restore multi-label targets afterwards.
    examples = prepare_chunks([dict(d, label=None) for d in docs], tok, cfg.strategy,
                              cfg.chunk_len, cfg.max_chunks, cfg.normalize_text, False)
    for ex, doc in zip(examples, docs):
        ex.pop("label")
        if "labels" in doc:
            ex["labels"] = doc["labels"]
    return examples
