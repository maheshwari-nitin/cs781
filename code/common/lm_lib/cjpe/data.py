"""CJPE data: [{"id", "text", "label"}] (train/dev); test has the same layout -- its labels are never read.

Judgments are far longer than the model context (median ~2.9k words), so each document is reduced to at most
`max_chunks` windows of `chunk_len` tokens according to a selection strategy:
  last        the final chunk_len * max_chunks tokens, split into chunks (reasoning + conclusion sit at the end)
  first_last  half the token budget from the beginning (facts, parties) and half from the end
  spread      chunks spaced evenly over the whole document (covers every section)
Only the parts that are needed are tokenised (character slices with a safety margin), so even the 737 MB training
file is prepared in minutes.
"""
import json
import random

import numpy as np

CHARS_PER_TOKEN_MARGIN = 8          # tokenise this many characters per needed token, then trim to the exact count


def load_split(path):
    d = json.load(open(path, encoding="utf-8"))
    items = d if isinstance(d, list) else [dict(id=k, **v) if isinstance(v, dict) else dict(id=k, text=v) for k, v in d.items()]
    return [dict(id=str(x["id"]), text=x["text"], label=x.get("label")) for x in items]


def normalize(text):
    return " ".join(text.split())


def subsample(docs, n, seed=1337):
    """Stratified random subset of n documents (keeps the label balance)."""
    if not n or n >= len(docs):
        return docs
    rng = random.Random(seed)
    by = {}
    for d in docs:
        by.setdefault(d["label"], []).append(d)
    out = []
    for lab, items in sorted(by.items(), key=lambda x: str(x[0])):
        rng.shuffle(items)
        out += items[:round(n * len(items) / len(docs))]
    rng.shuffle(out)
    return out


import re

OUTCOME_RE = re.compile(r"\b(appeals?|petitions?|writ petitions?)\b[^.]{0,60}\b(allowed|dismissed|disposed of)\b|"
                        r"\bwe (allow|dismiss)\b|\bare (allowed|dismissed)\b|\bis (allowed|dismissed)\b", re.I)


def drop_outcome_sentences(text, tail_chars=3000):
    """Remove sentences in the last `tail_chars` characters that state the outcome explicitly. The training split
    contains such sentences about twice as often as dev/test (17.6% vs 8.2%), which invites a shortcut."""
    head, tail = text[:-tail_chars], text[-tail_chars:]
    kept = [s for s in re.split(r"(?<=[.!?])\s+", tail) if not OUTCOME_RE.search(s)]
    return head + " ".join(kept)


def select_chunks(tok, text, strategy, chunk_len, max_chunks, do_normalize=True, drop_outcome=False):
    """-> list of token-id lists (1..max_chunks chunks, each <= chunk_len tokens)."""
    if do_normalize:
        text = normalize(text)
    if drop_outcome:
        text = drop_outcome_sentences(text)
    budget = chunk_len * max_chunks
    if strategy == "last":
        ids = tok.encode(text[-budget * CHARS_PER_TOKEN_MARGIN:]).ids[-budget:]
    elif strategy == "first_last":
        half = budget // 2
        head = tok.encode(text[:half * CHARS_PER_TOKEN_MARGIN]).ids[:half]
        tail = tok.encode(text[-half * CHARS_PER_TOKEN_MARGIN:]).ids[-half:]
        ids = head + tail if len(text) > budget * CHARS_PER_TOKEN_MARGIN else tok.encode(text).ids[-budget:]
    elif strategy == "spread":
        ids_all = tok.encode(text).ids
        if len(ids_all) <= budget:
            ids = ids_all
        else:
            starts = np.linspace(0, len(ids_all) - chunk_len, max_chunks).astype(int)
            return [ids_all[s:s + chunk_len] for s in starts]
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    ids = ids or [0]
    return [ids[i:i + chunk_len] for i in range(0, len(ids), chunk_len)][:max_chunks]


def prepare(docs, tok, strategy, chunk_len, max_chunks, do_normalize=True, drop_outcome=False, batch=2000):
    """Same result as select_chunks() per document, but tokenises with the multi-threaded encode_batch
    (all CPU cores), which matters for the 37k-document training split."""
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    budget = chunk_len * max_chunks
    half = budget // 2
    m = CHARS_PER_TOKEN_MARGIN
    out = []
    for i in range(0, len(docs), batch):
        part = docs[i:i + batch]
        texts = []
        for d in part:
            t = normalize(d["text"]) if do_normalize else d["text"]
            texts.append(drop_outcome_sentences(t) if drop_outcome else t)
        if strategy == "last":
            encs = tok.encode_batch([t[-budget * m:] for t in texts])
            chunk_ids = [e.ids[-budget:] for e in encs]
        elif strategy == "first_last":
            heads = tok.encode_batch([t[:half * m] for t in texts])
            tails = tok.encode_batch([t[-half * m:] for t in texts])
            chunk_ids = []
            for t, h, tl in zip(texts, heads, tails):
                chunk_ids.append(h.ids[:half] + tl.ids[-half:] if len(t) > budget * m else tok.encode(t).ids[-budget:])
        elif strategy == "spread":
            encs = tok.encode_batch(texts)
            chunk_ids = []
            for e in encs:
                ids = e.ids
                if len(ids) > budget:
                    starts = np.linspace(0, len(ids) - chunk_len, max_chunks).astype(int)
                    chunk_ids.append([x for s in starts for x in ids[s:s + chunk_len]])
                else:
                    chunk_ids.append(ids)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
        for d, ids in zip(part, chunk_ids):
            ids = ids or [0]
            out.append(dict(id=d["id"], label=d["label"],
                            chunks=[ids[j:j + chunk_len] for j in range(0, len(ids), chunk_len)][:max_chunks]))
    return out


def outcome_phrase_rate(docs, tail_chars=600):
    """Share of documents whose last `tail_chars` characters state the outcome explicitly ("appeal is allowed",
    "dismissed", ...). If train has this much more often than test, a model can learn the shortcut."""
    hits = sum(bool(OUTCOME_RE.search(d["text"][-tail_chars:])) for d in docs)
    return hits / max(len(docs), 1)
