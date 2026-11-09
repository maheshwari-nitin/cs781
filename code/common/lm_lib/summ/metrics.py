"""Reference-based evaluation only. BERTScore's encoder never supplies training targets."""


def score_summaries(predictions, references, bertscore=False, bert_model="roberta-large", bert_device="cpu"):
    if not predictions or set(predictions) != set(references):
        raise ValueError("Prediction/reference IDs must match exactly and be nonempty")
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    ids = sorted(references)
    hyp, ref = [predictions[i] for i in ids], [references[i] for i in ids]
    per = [scorer.score(r, p) for r, p in zip(ref, hyp)]
    result = {f"{key}_f1": sum(s[key].fmeasure for s in per) / len(per) for key in ("rouge1", "rougeL")}
    result["n"] = len(ids)
    if bertscore:
        from bert_score import BERTScorer
        # Deliberately short-lived: free the evaluation encoder after each evaluation.
        scorer = BERTScorer(model_type=bert_model, lang="en", device=bert_device)
        _, _, f = scorer.score(hyp, ref, batch_size=8)
        result["bertscore_f1"] = f.mean().item()
        result["bertscore_model"] = bert_model
        result["bertscore_hash"] = scorer.hash
    return result
