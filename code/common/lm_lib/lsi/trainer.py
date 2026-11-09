"""LSI training, Bernoulli distillation, saved dev calibration, and validated submissions.

Uses the existing DocClassifier, Phase-1 loaders and student-written adapters unchanged.
Runs are fresh fits; best.pt is weights for inference, not an optimizer-resume checkpoint.
"""
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..amp import autocast, describe, grad_scaler
from ..cjpe.model import DocClassifier
from ..cjpe.trainer import _batches
from ..ner.model import apply_method, enable_grad_checkpointing, param_report
from ..submit_json import write_json_submission
from ..trainer import load_trained
from . import data
from .metrics import calibrate, decode, multi_hot, scores


@dataclass
class LSIConfig:
    phase1_run: str = ""
    phase1_ckpt: str = "best"
    method: str = "full"
    method_kwargs: dict = field(default_factory=dict)
    data_dir: str = "/content/p5"
    out_dir: str = "/content/drive/MyDrive/cs781/lsi_runs"
    run_name: str = ""  # required: explicit names prevent accidental reuse
    label_ids: list = field(default_factory=list)  # optional official vocabulary; otherwise train/dev union
    strategy: str = "spread"
    chunk_len: int = 512
    max_chunks: int = 4
    token_pooling: str = "mean"
    chunk_agg: str = "mean"
    normalize_text: bool = True
    n_train: int = 0  # random smoke-test subset; label vocabulary still uses the full released splits
    batch_size: int = 2
    grad_accum: int = 4
    grad_checkpoint: bool = False
    epochs: int = 3
    lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    pos_weight: str = "none"  # none | balanced (train-only neg/pos, capped)
    pos_weight_cap: float = 20.0
    threshold: float = 0.5
    threshold_mode: str = "none"  # none | global | per_label; fit after checkpoint selection
    threshold_min_support: int = 5
    eval_every: int = 0
    kd_teacher: str = ""
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0
    stop_at: str = ""  # timezone-aware ISO timestamp; evaluate/save before returning
    seed: int = 1337

    def __post_init__(self):
        if not self.run_name or os.path.basename(self.run_name) != self.run_name or self.run_name in (".", ".."):
            raise ValueError("Provide a nonempty run_name (one directory name)")
        for name in ("chunk_len", "max_chunks", "batch_size", "grad_accum", "epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.strategy not in ("last", "first_last", "spread"):
            raise ValueError("Unknown chunk strategy")
        if self.token_pooling not in ("mean", "last") or self.chunk_agg not in ("mean", "max", "attention"):
            raise ValueError("Unknown pooling method")
        if self.pos_weight not in ("none", "balanced") or self.pos_weight_cap <= 0:
            raise ValueError("Invalid positive weighting")
        if self.threshold_mode not in ("none", "global", "per_label") or not 0 < self.threshold < 1:
            raise ValueError("Invalid threshold configuration")
        if self.n_train < 0 or self.eval_every < 0 or self.threshold_min_support < 1:
            raise ValueError("Invalid subset/evaluation/support setting")
        if not 0 <= self.kd_alpha <= 1 or self.kd_temperature <= 0:
            raise ValueError("Invalid distillation settings")
        if not 0 <= self.warmup_frac < 1:
            raise ValueError("warmup_frac must be in [0, 1)")


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_checkpoint(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def build_classifier(cfg, label_ids, device):
    backbone, tok, p1cfg, model_type = load_trained(cfg.phase1_run, cfg.phase1_ckpt, device=device)
    limit = p1cfg.src_len if model_type == "encdec" else p1cfg.seq_len
    if cfg.chunk_len > limit:
        raise ValueError(f"chunk_len={cfg.chunk_len} exceeds pretrained feature context {limit}")
    clf = DocClassifier(backbone.float(), len(label_ids), cfg.token_pooling, cfg.chunk_agg,
                        dropout=cfg.dropout).to(device)
    apply_method(clf, cfg.method, **cfg.method_kwargs)
    clf.to(device)
    # Probe actual token IDs: the vocab may be tiny in smoke tests. Freeze unused decoder parameters.
    was = clf.training
    clf.eval()
    ids = torch.zeros((1, 1, min(8, cfg.chunk_len)), dtype=torch.long, device=device)
    clf(ids, torch.ones_like(ids, dtype=torch.bool), torch.ones((1, 1), dtype=torch.bool, device=device)).sum().backward()
    for name, p in clf.named_parameters():
        if p.requires_grad and p.grad is None and ".experts." not in name:
            p.requires_grad_(False)
    clf.zero_grad(set_to_none=True)
    clf.train(was)
    if cfg.grad_checkpoint:
        enable_grad_checkpointing(clf.backbone)
    return clf, tok, param_report(clf)


def load_lsi_run(run_dir, device="cuda"):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        cfg = LSIConfig(**json.load(f))
    with open(os.path.join(run_dir, "label_ids.json"), encoding="utf-8") as f:
        label_ids = json.load(f)
    clf, tok, report = build_classifier(cfg, label_ids, device)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    if ck["label_ids"] != label_ids:
        raise ValueError("Checkpoint and saved label vocabulary disagree")
    clf.load_state_dict(ck["state"])
    return clf.eval(), tok, cfg, label_ids, np.asarray(ck["thresholds"]), report


@torch.no_grad()
def predict_probs(clf, examples, cfg, device, pad_id, bs=None):
    clf.eval()
    out = []
    for _, ids, tm, cm in _batches(examples, bs or cfg.batch_size, cfg.chunk_len, pad_id, False, None):
        with autocast(device):
            logits = clf(ids.to(device), tm.to(device), cm.to(device))
        out.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(out)


def _teacher(cfg, tok, label_ids, student_report, device):
    teacher, ttok, tcfg, tids, _, treport = load_lsi_run(cfg.kd_teacher, device)
    fields = ("strategy", "chunk_len", "max_chunks", "normalize_text")
    if any(getattr(cfg, f) != getattr(tcfg, f) for f in fields):
        raise ValueError("KD requires identical document selection and preprocessing")
    if json.loads(tok.to_str()) != json.loads(ttok.to_str()):
        raise ValueError("KD requires identical tokenizers; token IDs cannot be shared otherwise")
    if label_ids != tids:
        raise ValueError("KD requires identical statute vocabularies")
    if student_report["total_params"] >= treport["total_params"]:
        raise ValueError("Mandatory KD requires a smaller student; select a smaller Phase-1 backbone")
    teacher.requires_grad_(False)
    return teacher.eval(), treport


def _train_lsi(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    start = time.time()
    deadline = None
    if cfg.stop_at:
        from datetime import datetime
        dt = datetime.fromisoformat(cfg.stop_at)
        if dt.tzinfo is None:
            raise ValueError("stop_at must include a timezone")
        deadline = dt.timestamp()
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(f"Run already exists: {run_dir}. Use a new run_name; training does not resume.")
    train_all = data.load_split(os.path.join(cfg.data_dir, "train.json"))
    dev_docs = data.load_split(os.path.join(cfg.data_dir, "dev.json"))
    data.check_disjoint(train=train_all, dev=dev_docs)
    label_ids = data.label_vocab(train_all, dev_docs, cfg.label_ids or None)
    train_docs = (random.Random(cfg.seed).sample(train_all, cfg.n_train)
                  if 0 < cfg.n_train < len(train_all) else train_all)
    clf, tok, report = build_classifier(cfg, label_ids, device)
    teacher, teacher_report = _teacher(cfg, tok, label_ids, report, device) if cfg.kd_teacher else (None, None)
    pad_id = tok.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("Phase-1 tokenizer has no <|pad|> token")
    train, dev = data.prepare(train_docs, tok, cfg), data.prepare(dev_docs, tok, cfg)
    train_y, dev_y = multi_hot(train, label_ids), multi_hot(dev, label_ids)
    support = train_y.sum(0)
    unseen = [v for v, n in zip(label_ids, support) if n == 0]
    print(f"{cfg.run_name}: train={len(train)}, dev={len(dev)}, statutes={len(label_ids)}, "
          f"trainable={report['trainable_pct']:.2f}% | {describe(device)}", flush=True)
    if unseen:
        print("No positive training examples for these vocabulary IDs:", unseen)
    os.makedirs(run_dir, exist_ok=True)
    _dump(asdict(cfg), os.path.join(run_dir, "config.json"))
    _dump(label_ids, os.path.join(run_dir, "label_ids.json"))
    weight = None
    if cfg.pos_weight == "balanced":
        w = np.clip((len(train) - support) / np.maximum(support, 1), 1 / cfg.pos_weight_cap, cfg.pos_weight_cap)
        w[support == 0] = 1
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    body_ids = {id(p) for p in clf.backbone.parameters()}
    groups = [dict(params=[p for p in clf.parameters() if p.requires_grad and id(p) in body_ids], lr=cfg.lr),
              dict(params=[p for p in clf.parameters() if p.requires_grad and id(p) not in body_ids], lr=cfg.head_lr)]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    batches = math.ceil(len(train) / cfg.batch_size)
    total_steps = math.ceil(batches / cfg.grad_accum) * cfg.epochs
    warm = int(total_steps * cfg.warmup_frac)
    def lr_scale(s):
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total_steps - s) / max(1, total_steps - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = grad_scaler(device)
    params = [p for p in clf.parameters() if p.requires_grad]
    col = {v: i for i, v in enumerate(label_ids)}
    best, opt_steps, stopped = -1.0, 0, False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log_path = os.path.join(run_dir, "log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "opt_step", "train_loss", "dev_macro_f1", "dev_micro_f1", "dev_subset_accuracy", "seconds"])
        def evaluate(ep, loss):
            nonlocal best
            probs = predict_probs(clf, dev, cfg, device, pad_id)
            s = scores(dev_y, probs >= cfg.threshold, label_ids)
            writer.writerow([ep, opt_steps, loss, s["macro_f1"], s["micro_f1"], s["subset_accuracy"], time.time() - start])
            log.flush()
            print(f"\n[dev] epoch={ep} step={opt_steps} macro-F1={s['macro_f1']:.4f} micro-F1={s['micro_f1']:.4f}", flush=True)
            if s["macro_f1"] > best:
                best = s["macro_f1"]
                _save_checkpoint(dict(state=clf.state_dict(), label_ids=label_ids,
                                      thresholds=[cfg.threshold] * len(label_ids), epoch=ep,
                                      opt_step=opt_steps, scores=s), os.path.join(run_dir, "best.pt"))
                _dump(dict(ids=[d["id"] for d in dev], label_ids=label_ids, probs=probs.tolist()),
                      os.path.join(run_dir, "dev_probs.json"))
            clf.train()

        rng = random.Random(cfg.seed)
        for ep in range(1, cfg.epochs + 1):
            clf.train()
            opt.zero_grad(set_to_none=True)
            group_docs, loss_sum, n_docs = 0, 0.0, 0
            for n, (batch, ids, tm, cm) in enumerate(_batches(train, cfg.batch_size, cfg.chunk_len, pad_id, True, rng), 1):
                if deadline and time.time() >= deadline:
                    stopped = True
                    break
                y = torch.zeros((len(batch), len(label_ids)), device=device)
                for i, ex in enumerate(batch):
                    y[i, [col[v] for v in ex["labels"]]] = 1
                ids, tm, cm = ids.to(device), tm.to(device), cm.to(device)
                with autocast(device):
                    logits = clf(ids, tm, cm).float()
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=weight)
                if teacher is not None:
                    with torch.no_grad(), autocast(device):
                        target = (teacher(ids, tm, cm).float() / cfg.kd_temperature).sigmoid()
                    # Independent Bernoulli targets, not a softmax over mutually exclusive statutes.
                    kd = F.binary_cross_entropy_with_logits(logits / cfg.kd_temperature, target)
                    loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * cfg.kd_temperature ** 2 * kd
                scaler.scale(loss * len(batch)).backward()
                group_docs += len(batch)
                loss_sum += loss.item() * len(batch); n_docs += len(batch)
                if n % cfg.grad_accum == 0 or n == batches:
                    scaler.unscale_(opt)
                    for p in params:
                        if p.grad is not None:
                            p.grad.div_(group_docs)
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                    old_scale = scaler.get_scale()
                    scaler.step(opt); scaler.update()
                    if scaler.get_scale() >= old_scale:
                        sched.step(); opt_steps += 1
                        if hasattr(clf.backbone, "post_step"):
                            clf.backbone.post_step()
                    opt.zero_grad(set_to_none=True); group_docs = 0
                    if cfg.eval_every and opt_steps and opt_steps % cfg.eval_every == 0:
                        evaluate(ep, loss_sum / n_docs)
                if n == 1 or n % 20 == 0:
                    print(f"epoch {ep}/{cfg.epochs} batch {n}/{batches} loss {loss_sum / n_docs:.4f}", flush=True)
            # A deadline may interrupt an accumulation group. Discard its unstepped gradients.
            opt.zero_grad(set_to_none=True)
            evaluate(ep, loss_sum / max(n_docs, 1))
            if stopped:
                break
    with open(os.path.join(run_dir, "dev_probs.json"), encoding="utf-8") as f:
        saved = json.load(f)
    probs = np.asarray(saved["probs"])
    thresholds = calibrate(dev_y, probs, cfg.threshold_mode, cfg.threshold, cfg.threshold_min_support)
    fixed = scores(dev_y, probs >= cfg.threshold, label_ids)
    tuned = scores(dev_y, probs >= thresholds, label_ids)
    ck_path = os.path.join(run_dir, "best.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ck["thresholds"] = thresholds.tolist()
    _save_checkpoint(ck, ck_path)
    _dump(dict(mode=cfg.threshold_mode, label_ids=label_ids, thresholds=thresholds.tolist()),
          os.path.join(run_dir, "thresholds.json"))
    _dump(tuned["per_label"], os.path.join(run_dir, "per_label.json"))
    elapsed = time.time() - start
    results = dict(run=cfg.run_name, phase1_run=os.path.basename(cfg.phase1_run.rstrip("/\\")),
                   method=cfg.method, kd_teacher=cfg.kd_teacher, teacher_params=teacher_report,
                   strategy=cfg.strategy, chunk_len=cfg.chunk_len, max_chunks=cfg.max_chunks,
                   chunk_agg=cfg.chunk_agg, n_train=len(train), n_dev=len(dev), n_labels=len(label_ids),
                   train_unseen_labels=unseen, best_dev_macro_f1=best,
                   fixed_dev_scores=fixed, calibrated_dev_scores=tuned, threshold_mode=cfg.threshold_mode,
                   total_seconds=round(elapsed, 2), stopped_early=stopped, optimizer_steps=opt_steps,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None, **report)
    _dump(results, os.path.join(run_dir, "results.json"))
    return results


def train_lsi(cfg):
    try:
        return _train_lsi(cfg)
    except BaseException as e:
        from ..rr.trainer import free_gpu_after_failure
        free_gpu_after_failure(e)
        raise


def predict_test(run_dir, out_dir=None, device=None, bs=None, data_dir=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    clf, tok, cfg, label_ids, thresholds, _ = load_lsi_run(run_dir, device)
    # data_dir override supports inference after a fresh Colab session re-downloads the data.
    root = data_dir or cfg.data_dir
    docs = data.load_split(os.path.join(root, "test.json"), labeled=False)
    released = {split: data.load_split(os.path.join(root, f"{split}.json"), labeled=False)
                for split in ("train", "dev")}
    data.check_disjoint(**released, test=docs)
    examples = data.prepare(docs, tok, cfg)
    start = time.time()
    probs = predict_probs(clf, examples, cfg, device, tok.token_to_id("<|pad|>"), bs)
    pred = decode(probs, label_ids, thresholds)
    predictions = [dict(id=d["id"], pred=p) for d, p in zip(docs, pred)]
    if len(predictions) != len(docs) or any(len(p) != len(set(p)) or
            any(type(v) is not int or v not in label_ids for v in p) for p in pred):
        raise ValueError("Invalid LSI predictions")
    out_dir = out_dir or os.path.join(run_dir, "submission")
    zip_path = write_json_submission(predictions, {d["id"] for d in docs}, out_dir)
    _dump(dict(n_test=len(docs), inference_seconds=time.time() - start,
               mean_predicted_labels=float(np.mean([len(p) for p in pred])), empty_predictions=sum(not p for p in pred)),
          os.path.join(out_dir, "inference.json"))
    return zip_path, predictions
