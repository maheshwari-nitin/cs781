"""Multi-label metrics and dev-only calibration. Zero-division scores are zero."""
import numpy as np


def multi_hot(docs, label_ids):
    col = {v: i for i, v in enumerate(label_ids)}
    y = np.zeros((len(docs), len(label_ids)), dtype=np.float32)
    for i, doc in enumerate(docs):
        for v in doc["labels"]:
            y[i, col[v]] = 1
    return y


def scores(gold, pred, label_ids=None):
    gold, pred = np.asarray(gold, dtype=bool), np.asarray(pred, dtype=bool)
    if gold.shape != pred.shape or gold.ndim != 2 or not all(gold.shape):
        raise ValueError("Expected nonempty, matching [documents, labels] matrices")
    tp = (gold & pred).sum(0)
    fp, fn = (~gold & pred).sum(0), (gold & ~pred).sum(0)
    def ratio(n, d):
        return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)
    p, r, f = ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(2 * tp, 2 * tp + fp + fn)
    denom = int(2 * tp.sum() + fp.sum() + fn.sum())
    ids = list(range(gold.shape[1])) if label_ids is None else label_ids
    return dict(macro_f1=float(f.mean()), micro_f1=float(2 * tp.sum() / denom) if denom else 0.0,
                subset_accuracy=float((gold == pred).all(1).mean()),
                macro_precision=float(p.mean()), macro_recall=float(r.mean()),
                per_label=[dict(label=int(v), support=int(gold[:, i].sum()), precision=float(p[i]),
                                recall=float(r[i]), f1=float(f[i])) for i, v in enumerate(ids)])


def calibrate(gold, probs, mode="none", threshold=0.5, min_support=5):
    """Coarse dev grid; unsupported rare labels retain the global threshold.

    Per-label calibration is optimistic on the same dev set; record both fixed and tuned scores.
    """
    gold, probs = np.asarray(gold), np.asarray(probs)
    if mode not in ("none", "global", "per_label"):
        raise ValueError(f"Unknown threshold mode: {mode}")
    thresholds = np.full(probs.shape[1], threshold, dtype=float)
    grid = sorted(set([threshold] + np.linspace(0.05, 0.95, 19).round(3).tolist()))
    if mode == "none":
        return thresholds
    best = scores(gold, probs >= thresholds)["macro_f1"]
    for t in grid:
        f = scores(gold, probs >= t)["macro_f1"]
        if f > best:
            best, thresholds[:] = f, t
    if mode == "per_label":
        for j in range(probs.shape[1]):
            if gold[:, j].sum() < min_support or (1 - gold[:, j]).sum() < min_support:
                continue
            y, pr = gold[:, j:j + 1], probs[:, j:j + 1]
            best = scores(y, pr >= thresholds[j])["macro_f1"]
            for t in grid:
                f = scores(y, pr >= t)["macro_f1"]
                if f > best:
                    best, thresholds[j] = f, t
    return thresholds


def decode(probs, label_ids, thresholds):
    probs = np.asarray(probs)
    if probs.ndim != 2 or probs.shape[1] != len(label_ids) or not np.isfinite(probs).all():
        raise ValueError("Invalid probability matrix")
    return [[int(v) for v, keep in zip(label_ids, row) if keep] for row in probs >= thresholds]
