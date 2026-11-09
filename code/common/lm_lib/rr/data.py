"""RR data: documents are lists of sentences (+ one label per sentence for train/dev).

Train/dev files: [{"id", "text": [sentences], "labels": [labels]}, ...]; test files: {doc_id: [sentences]}.
Each document is tokenised as one stream (sentence i encoded with a leading space), recording each sentence's
token range, so sentences can be pooled from a forward pass over a window that also contains their neighbours.
"""
import json
import os
from collections import Counter

import numpy as np

from ..ner.data import make_windows

DOMAINS = ("CL", "IT")


def load_labels(path):
    vocab = json.load(open(path, encoding="utf-8"))
    labels = [l for l, _ in sorted(vocab.items(), key=lambda x: x[1])]
    return labels, {l: i for i, l in enumerate(labels)}


def load_docs(data_dir, split, domains=DOMAINS):
    """split: train | dev | test -> list of dict(id, domain, text, labels or None)."""
    docs = []
    for dom in domains:
        d = json.load(open(os.path.join(data_dir, f"{dom}_{split}.json"), encoding="utf-8"))
        items = d if isinstance(d, list) else [{"id": k, "text": v} for k, v in d.items()]
        for x in items:
            docs.append(dict(id=str(x["id"]), domain=dom, text=list(x["text"]), labels=x.get("labels")))
    return docs


def tokenize_doc(tok, sents, max_sent_tokens, pad_id):
    ids, bounds = [], []
    for i, s in enumerate(sents):
        e = tok.encode((" " if i else "") + s).ids[:max_sent_tokens] or [pad_id]
        bounds.append((len(ids), len(ids) + len(e)))
        ids += e
    return np.asarray(ids, dtype=np.int64), bounds


class RRDataset:
    """context='document': overlapping token windows over the whole document; every sentence fully inside a window
    is a training target for that window, and at prediction time each sentence is read from one owner window
    (most left context for causal models, most central for bidirectional encoders).
    context='sentence': every sentence is its own example (no surrounding context)."""

    def __init__(self, docs, tok, label2id, window, stride, context="document", causal=True, max_sent_tokens=256):
        pad_id = tok.token_to_id("<|pad|>")
        max_sent_tokens = min(max_sent_tokens, window - stride) if context == "document" else min(max_sent_tokens, window)
        self.docs, self.train_examples, self.pred_examples = {}, [], []
        for doc in docs:
            ids, bounds = tokenize_doc(tok, doc["text"], max_sent_tokens, pad_id)
            lab = [label2id[l] for l in doc["labels"]] if doc["labels"] else None
            self.docs[doc["id"]] = dict(ids=ids, bounds=bounds, labels=lab, domain=doc["domain"], n=len(doc["text"]))
            if context == "sentence":
                for i, (s, e) in enumerate(bounds):
                    ex = (doc["id"], s, e, [(i, 0, e - s)])
                    self.train_examples.append(ex); self.pred_examples.append(ex)
                continue
            wins = make_windows(len(ids), window, stride)
            inside = [[(i, s - a, e - a) for i, (s, e) in enumerate(bounds) if s >= a and e <= z] for a, z in wins]
            owner = {}
            for w, (a, z) in enumerate(wins):
                for i, s, e in inside[w]:
                    score = s if causal else min(s, (z - a) - e)        # left context / distance to nearest edge
                    if i not in owner or score > owner[i][0]:
                        owner[i] = (score, w)
            assert len(owner) == len(bounds), "a sentence fits in no window (max_sent_tokens must be <= window - stride)"
            for w, (a, z) in enumerate(wins):
                if inside[w]:
                    self.train_examples.append((doc["id"], a, z, inside[w]))
                own = [x for x in inside[w] if owner[x[0]][1] == w]
                if own:
                    self.pred_examples.append((doc["id"], a, z, own))

    def label_counts(self, n_labels):
        c = Counter(l for d in self.docs.values() if d["labels"] for l in d["labels"])
        return np.array([c.get(i, 0) for i in range(n_labels)], dtype=np.int64)
