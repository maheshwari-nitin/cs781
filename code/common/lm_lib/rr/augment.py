"""Label-preserving data augmentation for RR, built only from the official training documents (no external data,
no LLM-generated text).

Rhetorical roles come in contiguous segments with a strong positional order (Fact early, RulingByPresentCourt last),
so synthetic documents are made by *segment swapping*: take a real training document's sequence of role segments as
a template and replace every segment with a real segment of the same role drawn from another training document of
the same domain. All sentences and labels are genuine; the discourse structure stays realistic; rare roles get seen
in many new contexts. Templates that contain rare roles can be up-weighted.

Limits (state them in the report): roles with very few segments (e.g. Dissent: 2 segments, PrecedentOverruled: 2)
cannot gain real diversity this way -- recombination cannot invent new examples of them.
"""
import random
from collections import Counter, defaultdict
from itertools import groupby


def segments(doc):
    """[(label, [sentences])] contiguous runs of the same role."""
    out, i = [], 0
    for lab, g in groupby(doc["labels"]):
        n = len(list(g))
        out.append((lab, doc["text"][i:i + n]))
        i += n
    return out


def rare_roles(docs, threshold):
    c = Counter(l for d in docs for l in d["labels"])
    return {l for l, n in c.items() if n < threshold}


def segment_swap_docs(docs, n_per_domain, rare, rare_boost=3.0, seed=1337):
    """Synthetic documents (same dict format as rd.load_docs) from segment swapping within each domain."""
    rng = random.Random(seed)
    out = []
    for dom in sorted({d["domain"] for d in docs}):
        dd = [d for d in docs if d["domain"] == dom and d["labels"]]
        pool = defaultdict(list)                       # role -> [(source doc id, sentences)]
        for d in dd:
            for lab, sents in segments(d):
                pool[lab].append((d["id"], sents))
        weights = [rare_boost if rare & set(d["labels"]) else 1.0 for d in dd]
        for k in range(n_per_domain):
            tpl = rng.choices(dd, weights=weights)[0]
            text, labels = [], []
            for lab, sents in segments(tpl):
                cands = [s for src, s in pool[lab] if src != tpl["id"]] or [sents]
                seg = rng.choice(cands)
                text += seg
                labels += [lab] * len(seg)
            out.append(dict(id=f"synth_{dom}_{k}", domain=dom, text=text, labels=labels))
    return out


def oversample_rare_windows(examples, docs_by_id, rare_ids, copies):
    """Repeat training windows that contain at least one rare-role sentence `copies` extra times."""
    if copies <= 0 or not rare_ids:
        return list(examples)
    extra = []
    for ex in examples:
        d, _, _, sents = ex
        labs = docs_by_id[d]["labels"]
        if any(labs[i] in rare_ids for i, _, _ in sents):
            extra += [ex] * copies
    return list(examples) + extra


def summary(train_docs, synth_docs):
    real = Counter(l for d in train_docs for l in d["labels"])
    syn = Counter(l for d in synth_docs for l in d["labels"])
    return {l: dict(real=real[l], synthetic=syn.get(l, 0)) for l in sorted(real, key=lambda x: real[x])}
