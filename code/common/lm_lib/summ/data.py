"""Phase-6 schema, reproducible document holdout and exact token-budget preparation."""
import hashlib
import json
import random
import warnings


def load_split(path, labeled=True, *, skip_empty_summaries=True, report=None):
    """Load valid records; report missing/blank training references without dropping test cases.

    Unexpected summary types still fail to expose schema mismatches. Set
    skip_empty_summaries=False for strict auditing of missing/blank references.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, (list, dict)):
        raise ValueError("Expected a list or ID-to-record object")
    items = raw if isinstance(raw, list) else [dict(v, id=k) for k, v in raw.items()]
    docs, seen, skipped = [], set(), []
    for record in items:
        ident = str(record["id"])
        if ident in seen:
            raise ValueError(f"Duplicate document ID: {ident}")
        seen.add(ident)
        text = record.get("document")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{ident}: expected nonempty 'document' string (Phase 6 schema)")
        doc = dict(id=ident, document=text)
        if labeled:
            summary = record.get("summary")
            if summary is None or (isinstance(summary, str) and not summary.strip()):
                if not skip_empty_summaries:
                    raise ValueError(f"{ident}: expected nonempty reference summary")
                skipped.append(ident)
                continue
            if not isinstance(summary, str):
                raise ValueError(f"{ident}: expected reference summary string, got {type(summary).__name__}")
            doc["summary"] = summary
        # Test reference fields are never accessed, even if present.
        docs.append(doc)
    if report is not None:
        report.update(total_records=len(items), loaded_records=len(docs),
                      skipped_empty_summary_ids=skipped)
    if skipped:
        warnings.warn(f"{path}: skipped {len(skipped)} training records with missing/blank summaries "
                      f"(IDs: {', '.join(skipped[:10])}); kept {len(docs)} records", UserWarning, stacklevel=2)
    if not docs:
        raise ValueError(f"No usable records in split: {path}; check its schema and summaries")
    return docs


def fingerprint(docs):
    """Order-independent fingerprint of normalized loaded records, including train references."""
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d["id"]):
        h.update(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def split_train_val(docs, val_fraction=0.05, seed=1337):
    """Hold out duplicate normalized documents together to prevent text leakage across splits."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    groups = {}
    for doc in docs:
        key = hashlib.sha256(" ".join(doc["document"].split()).encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(doc)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("Need at least two distinct documents for a held-out validation split")
    random.Random(seed).shuffle(keys)
    nval = min(len(keys) - 1, max(1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:nval])
    train = [d for k in keys if k not in val_keys for d in sorted(groups[k], key=lambda d: d["id"])]
    val = [d for k in keys if k in val_keys for d in sorted(groups[k], key=lambda d: d["id"])]
    return train, val


def select_tokens(ids, budget, strategy):
    if budget < 1:
        raise ValueError("Document token budget must be positive")
    if strategy not in ("first", "last", "first_last"):
        raise ValueError(f"Unknown context selection: {strategy}")
    if len(ids) <= budget:
        return ids
    if strategy == "first":
        return ids[:budget]
    if strategy == "last":
        return ids[-budget:]
    head = (budget + 1) // 2
    return ids[:head] + ids[-(budget - head):] if budget > head else ids[:head]


def source_ids(doc, tok, cfg):
    text = " ".join(doc["document"].split()) if cfg.normalize_text else doc["document"]
    prefix = tok.encode("Legal document:\n").ids
    suffix = tok.encode("\n\nSummary:\n").ids
    budget = cfg.source_tokens - len(prefix) - len(suffix) - 1
    ids = select_tokens(tok.encode(text).ids, budget, cfg.strategy)
    return [tok.token_to_id("<|bos|>")] + prefix + ids + suffix


def prepare(doc, tok, cfg, model_type, summary=None):
    """Decoder labels mask the prompt; encoder-decoder labels are summary tokens only."""
    src = source_ids(doc, tok, cfg)
    target = tok.encode(doc["summary"] if summary is None else summary).ids[:cfg.target_tokens - 1]
    if not target:
        raise ValueError(f"{doc['id']}: summary encodes to zero tokens")
    target += [tok.token_to_id("<|endoftext|>")]
    if model_type == "encdec":
        return dict(id=doc["id"], src=src,
                    x=[tok.token_to_id("<|bos|>")] + target[:-1], y=target)
    seq = src + target
    return dict(id=doc["id"], x=seq[:-1], y=[-100] * (len(src) - 1) + target)
