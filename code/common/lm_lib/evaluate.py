"""Local approximation of the Codabench metrics on our own held-out prompts.
BERTScore uses roberta-large (the model the official scorer reports needing); exact settings of
the official scorer (e.g. baseline rescaling) are unknown, so compare runs relative to each other."""

_SCORER = None


def get_bertscorer():
    """Load roberta-large once and reuse it (bert_score.score() reloads the model on every call)."""
    global _SCORER
    if _SCORER is None:
        from bert_score import BERTScorer
        _SCORER = BERTScorer(model_type="roberta-large", lang="en")
    return _SCORER


def local_scores(preds, refs, bertscore=True, batch_size=32, verbose=True):
    ids = [i for i in refs if i in preds]
    hyp = [preds[i] for i in ids]
    ref = [refs[i]["text"] if isinstance(refs[i], dict) else refs[i] for i in ids]
    out = {"n": len(ids)}

    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    out["rougeL_f1"] = sum(sc.score(r, h)["rougeL"].fmeasure for r, h in zip(ref, hyp)) / len(ids)

    import sacrebleu
    out["bleu4"] = sacrebleu.corpus_bleu(hyp, [ref]).score

    if bertscore:
        _, _, f1 = get_bertscorer().score(hyp, ref, batch_size=batch_size)
        out["bertscore_f1"] = f1.mean().item()
    if verbose:
        for k, v in out.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    return out
