"""Explicit A experiment configurations and measured-dev champion selection.

These configurations are candidates, not claims of measured improvement. Training executes in Colab.
"""
import math
import os
from dataclasses import replace

from .trainer import PCRConfig, read


def a_more_training_config(base, run_name="A_full__more_steps_v2"):
    if base.method != "full" or base.kd_teacher:
        raise ValueError("A training-budget candidate requires the undistilled full-fine-tuning configuration")
    effective = base.effective_batch_queries or base.train_batch_size * base.grad_accum
    return replace(base, run_name=run_name, epochs=max(30, base.epochs * 4),
                   lr=3e-5, warmup_frac=0.03, early_stopping_patience=6,
                   negative_sampling="random", resample_negatives=True,
                   eval_every=0, auto_batch=True, batch_calibration={},
                   train_batch_size=1, grad_accum=effective,
                   effective_batch_queries=effective, max_train_batch=effective)


def a_tuning_configs(base):
    longer = a_more_training_config(base)
    hard = replace(longer, run_name="A_full__hard12_v2", negative_sampling="tfidf",
                   negatives=12, max_positives=4, hard_negative_fraction=0.5,
                   hard_negative_pool_size=64)
    context = replace(hard, run_name="A_full__long4x512_v2", chunk_len=512, max_chunks=4,
                      strategy="spread", token_pooling="last",
                      effective_batch_queries=16, train_batch_size=1, grad_accum=16, max_train_batch=16)
    return dict(more_steps=longer, hard_negatives=hard, long_context=context)


def select_best_a_run(baseline_dir, candidate_dirs):
    """Keep baseline on ties/regressions; refuse scores from incompatible datasets/query subsets."""
    base_cfg = PCRConfig(**read(os.path.join(baseline_dir, "config.json")))
    base_manifest = read(os.path.join(baseline_dir, "manifest.json"))
    baseline_effective = base_cfg.effective_batch_queries or base_cfg.train_batch_size * base_cfg.grad_accum
    rows = []
    for run_dir in [baseline_dir] + list(candidate_dirs):
        cfg = PCRConfig(**read(os.path.join(run_dir, "config.json")))
        manifest = read(os.path.join(run_dir, "manifest.json"))
        result = read(os.path.join(run_dir, "results.json"))
        if (cfg.phase1_run != base_cfg.phase1_run or cfg.method != "full" or cfg.kd_teacher or
                manifest["phase1_sha256"] != base_manifest["phase1_sha256"]):
            raise ValueError("A comparison must use the same Phase-1 backbone/checkpoint and full adaptation")
        if (manifest["train_fingerprint"] != base_manifest["train_fingerprint"] or
                manifest["dev_fingerprint"] != base_manifest["dev_fingerprint"] or
                set(manifest["eval_query_ids"]) != set(base_manifest["eval_query_ids"]) or
                set(manifest["used_train_query_ids"]) != set(base_manifest["used_train_query_ids"])):
            raise ValueError("A comparison needs identical train data/query subset and dev data/query subset")
        score = float(result["best_dev_micro_f1_at7"])
        if not math.isfinite(score) or not os.path.isfile(os.path.join(run_dir, "best.pt")):
            raise ValueError("Candidate needs finite measured metrics and best.pt")
        rows.append(dict(run_dir=str(run_dir), run=cfg.run_name, micro_f1_at7=score,
                         epochs_budget=cfg.epochs, completed_epochs=result.get("completed_epochs"),
                         best_epoch=result.get("best_epoch"), optimizer_steps=result.get("optimizer_steps"),
                         lr=cfg.lr, negatives=cfg.negatives, negative_sampling=cfg.negative_sampling,
                         source_token_budget=cfg.chunk_len * cfg.max_chunks,
                         effective_batch=result.get("effective_batch_queries", baseline_effective)))
    champion = max(rows, key=lambda r: r["micro_f1_at7"])
    return dict(baseline_run=str(baseline_dir), baseline_score=rows[0]["micro_f1_at7"],
                selected_run=champion["run_dir"], selected_score=champion["micro_f1_at7"],
                delta=champion["micro_f1_at7"] - rows[0]["micro_f1_at7"], comparisons=rows,
                selection_basis="same official dev queries/pool; dev-tuned score is not an unbiased held-out estimate")
