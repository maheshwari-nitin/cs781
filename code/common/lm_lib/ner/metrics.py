"""Strict span scoring: a prediction counts only if (doc, start, end, label) matches a gold entity exactly.
Local approximation of the official scorer; macro averages over labels that occur in gold or predictions.
If you have the official evaluation script, prefer it for final numbers."""
from collections import defaultdict


def strict_scores(pred, gold, labels):
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for d in gold.keys() | pred.keys():
        g = {(e[0], e[1], e[2]) for e in gold.get(d, [])}
        p = {(e[0], e[1], e[2]) for e in pred.get(d, [])}
        for e in p & g:
            tp[e[2]] += 1
        for e in p - g:
            fp[e[2]] += 1
        for e in g - p:
            fn[e[2]] += 1
    per = {}
    for l in labels:
        if tp[l] + fp[l] + fn[l] == 0:
            continue
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[l] = dict(p=P, r=R, f1=F, tp=tp[l], fp=fp[l], fn=fn[l])
    k = max(len(per), 1)
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = T / (T + FP) if T + FP else 0.0
    micro_r = T / (T + FN) if T + FN else 0.0
    return dict(macro_f1=sum(v["f1"] for v in per.values()) / k,
                macro_p=sum(v["p"] for v in per.values()) / k,
                macro_r=sum(v["r"] for v in per.values()) / k,
                micro_f1=2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0,
                per_label=per)


def format_scores(s):
    lines = [f"strict macro-F1 {s['macro_f1']:.4f} | macro-P {s['macro_p']:.4f} | macro-R {s['macro_r']:.4f} | "
             f"micro-F1 {s['micro_f1']:.4f}"]
    for l, v in sorted(s["per_label"].items(), key=lambda x: -x[1]["f1"]):
        lines.append(f"  {l:10s} F1 {v['f1']:.3f}  P {v['p']:.3f}  R {v['r']:.3f}  (tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
    return "\n".join(lines)
