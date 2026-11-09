"""L-NER data: loading, character<->token alignment, BIO labels, windowing and span decoding.

Gold format: {doc_id: [[start, end, label, text], ...]} with `end` exclusive (text[start:end] == entity text).
Tokens come from the Phase-1 byte-level BPE; its offsets include the leading space of "Ġword" tokens, so
token spans are trimmed of leading whitespace before alignment and decoding.
"""
import bisect
import json
import os
import random
from collections import Counter

import numpy as np


def load_labels(path):
    labels = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    tags = ["O"] + [f"{p}-{l}" for l in labels for p in ("B", "I")]
    return labels, tags, {t: i for i, t in enumerate(tags)}


def load_folds(data_dir, folds):
    texts, golds = {}, {}
    for k in folds:
        texts.update(json.load(open(os.path.join(data_dir, f"fold{k}-text.json"), encoding="utf-8")))
        gp = os.path.join(data_dir, f"fold{k}-gold.json")
        if os.path.exists(gp):
            golds.update(json.load(open(gp, encoding="utf-8")))
    return texts, golds


def check_gold(texts, golds):
    """Verify offsets against the entity text and look for overlaps (BIO cannot represent overlapping spans)."""
    bad, overlaps, n = [], 0, 0
    for d, ents in golds.items():
        prev_end = -1
        for e in sorted(ents, key=lambda x: (x[0], x[1])):
            n += 1
            if len(e) > 3 and texts[d][e[0]:e[1]] != e[3]:
                bad.append((d, e))
            if e[0] < prev_end:
                overlaps += 1
            prev_end = max(prev_end, e[1])
    return dict(entities=n, offset_mismatches=len(bad), examples=bad[:5], overlapping=overlaps)


def tokenize_doc(tok, text):
    """-> (ids, spans) where spans[i] = (start, end) char offsets of token i, leading whitespace trimmed."""
    enc = tok.encode(text)
    spans = []
    for s, e in enc.offsets:
        while s < e and text[s].isspace():
            s += 1
        spans.append((s, e))
    return list(enc.ids), spans


def bio_labels(spans, ents, tag2id):
    """Token BIO ids for one doc + alignment stats (entities whose boundaries fall exactly on token boundaries)."""
    n = len(spans)
    lab = np.zeros(n, dtype=np.int64)
    ends = [e for _, e in spans]
    exact = unaligned = 0
    for s, e, label, *_ in sorted(ents, key=lambda x: (x[0], x[1])):
        i = bisect.bisect_right(ends, s)          # first token ending after s
        toks = []
        while i < n and spans[i][0] < e:
            nonempty = spans[i][1] > spans[i][0]
            # whitespace-only tokens strictly inside the entity (e.g. a double space) get I-X too; leaving them
            # "O" would split the entity in two (O -> I-X is not valid BIO)
            if (nonempty and spans[i][1] > s) or (not nonempty and toks and spans[i][1] < e):
                toks.append(i)
            i += 1
        if not toks:
            unaligned += 1
            continue
        lab[toks[0]] = tag2id[f"B-{label}"]
        for t in toks[1:]:
            lab[t] = tag2id[f"I-{label}"]
        exact += spans[toks[0]][0] == s and spans[toks[-1]][1] == e
    return lab, dict(exact=exact, unaligned=unaligned, total=len(ents))


def make_windows(n, window, stride):
    if n == 0:
        return []                     # empty document (fold 3 has one): no windows, predicted as []
    if n <= window:
        return [(0, n)]
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    return [(a, a + window) for a in starts]


def token_owner(windows, n, causal):
    """For each token, the window whose prediction is used: most left context (causal models) or most
    central position (bidirectional encoders)."""
    best = np.full(n, -1, dtype=np.int64)
    score = np.full(n, -1, dtype=np.int64)
    for w, (a, b) in enumerate(windows):
        t = np.arange(a, b)
        s = (t - a) if causal else np.minimum(t - a, b - 1 - t)
        upd = s > score[a:b]
        score[a:b][upd] = s[upd]
        best[a:b][upd] = w
    return best


def allowed_transitions(tags):
    """BIO validity: I-X may only follow B-X or I-X; everything else may follow anything.
    Returns (trans [n, n] bool: prev -> cur allowed, start [n] bool: allowed as the first tag)."""
    n = len(tags)
    trans = np.ones((n, n), dtype=bool)
    start = np.ones(n, dtype=bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            lab = t[2:]
            start[j] = False
            for i, p in enumerate(tags):
                trans[i, j] = p in (f"B-{lab}", f"I-{lab}")
    return trans, start


def viterbi_bio(logp, tags):
    """Most probable tag sequence that is valid BIO, given per-token log-probabilities [T, n_tags]."""
    trans, start = allowed_transitions(tags)
    T, n = logp.shape
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    pen = np.where(trans, 0.0, -1e9)
    score = np.where(start, logp[0], -1e9)
    back = np.zeros((T, n), dtype=np.int64)
    for t in range(1, T):
        cand = score[:, None] + pen                      # [prev, cur]
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# BPE often merges trailing punctuation into an entity's last token (e.g. "SCC 47]," or "Ors.["); in the training
# gold, entities essentially never end with these characters (2 of 8393 end with ","), so they are trimmed.
TRIM_END = " \t\n,;:[(\"'"


def bio_to_spans(pred, spans, text, tags, trim_end=TRIM_END):
    """Token tag ids -> [[start, end, label], ...]. An I- tag without a matching open entity starts a new one."""
    out, cur = [], None

    def close():
        if cur is not None:
            s, e, l = cur
            e = s + len(text[s:e].rstrip(trim_end))
            if e > s:
                out.append([s, e, l])

    for i, p in enumerate(pred):
        if spans[i][1] <= spans[i][0]:            # whitespace-only token: transparent (never opens or closes)
            continue
        tag = tags[int(p)]
        if tag == "O":
            close(); cur = None
            continue
        bi, label = tag.split("-", 1)
        if bi == "B" or cur is None or cur[2] != label:
            close(); cur = [spans[i][0], spans[i][1], label]
        else:
            cur[1] = spans[i][1]
    close()
    return out


class NERDataset:
    """Tokenised docs + fixed-length windows (training examples)."""

    def __init__(self, texts, golds, tok, tag2id, window, stride):
        self.docs = {}
        self.examples = []      # (doc_id, a, b)
        stats = Counter()
        for d, text in texts.items():
            ids, spans = tokenize_doc(tok, text)
            lab, st = bio_labels(spans, golds.get(d, []), tag2id) if golds else (None, {})
            stats.update(st)
            wins = make_windows(len(ids), window, stride)
            self.docs[d] = dict(ids=np.asarray(ids, dtype=np.int64), spans=spans, labels=lab, windows=wins)
            self.examples += [(d, a, b) for a, b in wins]
        self.align_stats = dict(stats)

    def label_counts(self, n_tags):
        c = np.zeros(n_tags, dtype=np.int64)
        for doc in self.docs.values():
            if doc["labels"] is not None:
                c += np.bincount(doc["labels"], minlength=n_tags)
        return c


def split_docs(texts, golds, n_val, seed=1337):
    ids = sorted(texts)
    random.Random(seed).shuffle(ids)
    val = set(ids[:n_val])
    pick = lambda keep: ({d: texts[d] for d in ids if keep(d)}, {d: golds[d] for d in ids if keep(d) and d in golds})
    return pick(lambda d: d not in val), pick(lambda d: d in val)
