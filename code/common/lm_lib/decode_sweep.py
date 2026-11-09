"""Evaluate a grid of decoding settings (temperature / top-k / top-p / repetition penalty) on the held-out
validation prompts -- never on the official test prompts.

Every setting is run on the same fixed prompt subset; sampled settings (temperature > 0) are repeated over
several seeds and reported as mean +- std, so differences can be told apart from sampling noise.
Besides BERTScore / ROUGE-L / BLEU-4 it records diagnostics that help explain *why* a setting scores
the way it does: repeated 4-grams (degenerate loops), distinct-2 (diversity), output length.
Results are appended to a JSONL file; settings already in the file are skipped, so a sweep can resume.
"""
import json
import os
import random
import time
from collections import Counter

import numpy as np

from .evaluate import local_scores
from . import generate as _gen
from .generate import generate_continuations


def _key(s, gen=None):
    # generation settings (token cap, banned special tokens) are part of a result's identity, so changing them
    # re-runs the setting instead of silently re-using an old result
    return json.dumps(dict(sampling=s, gen=gen), sort_keys=True)


def text_diagnostics(preds, n_words=128):
    rep4, dist2, lens = [], [], []
    for t in preds.values():
        w = t.split()
        lens.append(len(w))
        g4 = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
        rep4.append(1 - len(set(g4)) / len(g4) if g4 else 0.0)
        g2 = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
        dist2.append(len(set(g2)) / len(g2) if g2 else 0.0)
    return dict(repeat_4gram=float(np.mean(rep4)), distinct_2=float(np.mean(dist2)),
                avg_words=float(np.mean(lens)), frac_short=float(np.mean([l < n_words for l in lens])))


def run_sweep(model, tok, mtype, cfg, prompts, refs, grid, out_path, n_prompts=200, seeds=(0, 1, 2),
              gen=None, prompt_seed=0):
    """grid: list of sampling dicts, e.g. {'temperature': 0.7, 'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.1}.
    Returns a list of result dicts (also appended to out_path)."""
    gen = dict(dict(n_words=128, max_new_tokens=256, batch_size=64, ban_special=True), **(gen or {}))
    gen_id = {k: v for k, v in gen.items() if k != "batch_size"}   # batch size doesn't change outputs
    ids = sorted(prompts)
    random.Random(prompt_seed).shuffle(ids)
    ids = sorted(ids[:n_prompts], key=int) if all(i.isdigit() for i in ids) else ids[:n_prompts]
    sub_p = {i: prompts[i] for i in ids}
    sub_r = {i: refs[i] for i in ids}
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done[_key(r["sampling"], r.get("gen"))] = r
    results = []
    for s in grid:
        s = dict(dict(temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0), **s)
        if _key(s, gen_id) in done:
            results.append(done[_key(s, gen_id)])
            print(f"skip (done): {s}")
            continue
        run_seeds = list(seeds) if s["temperature"] > 0 else [0]   # greedy is deterministic
        per_seed, t0 = [], time.time()
        for sd in run_seeds:
            preds = generate_continuations(model, tok, mtype, sub_p, cfg.seq_len, cfg.src_len, sampling=s, seed=sd,
                                           device=next(model.parameters()).device.type, **gen)
            m = local_scores(preds, sub_r, verbose=False)
            m.update(text_diagnostics(preds, gen["n_words"]))
            st = _gen.LAST_STATS
            m.update(frac_short_eot=st["short_eot"] / st["n"], frac_short_cap=st["short_token_cap"] / st["n"])
            per_seed.append(m)
        keys = [k for k in per_seed[0] if k != "n"]
        r = dict(sampling=s, gen=gen_id, n_prompts=len(ids), seeds=run_seeds, seconds=round(time.time() - t0, 1))
        for k in keys:
            vals = [m[k] for m in per_seed]
            r[k] = float(np.mean(vals))
            r[k + "_std"] = float(np.std(vals))
        with open(out_path, "a") as f:
            f.write(json.dumps(r) + "\n")
        results.append(r)
        print(f"{s} -> BERTScore {r['bertscore_f1']:.4f} (+-{r['bertscore_f1_std']:.4f}) | ROUGE-L {r['rougeL_f1']:.4f} | "
              f"rep4 {r['repeat_4gram']:.3f} | distinct2 {r['distinct_2']:.3f} | {r['seconds']:.0f}s")
    return results


def results_table(path):
    import pandas as pd
    rows = []
    for line in open(path):
        r = json.loads(line)
        g = r.get("gen") or {}
        rows.append(dict(**r["sampling"], ban_special=g.get("ban_special", False), max_new_tokens=g.get("max_new_tokens", 256),
                         **{k: v for k, v in r.items() if k not in ("sampling", "gen")}))
    df = pd.DataFrame(rows)
    cols = ["temperature", "top_k", "top_p", "repetition_penalty", "ban_special", "max_new_tokens", "bertscore_f1",
            "bertscore_f1_std", "rougeL_f1", "bleu4", "repeat_4gram", "distinct_2", "avg_words", "frac_short",
            "frac_short_eot", "frac_short_cap", "seconds"]
    return df[[c for c in cols if c in df]].sort_values("bertscore_f1", ascending=False).reset_index(drop=True)
