"""Train / evaluate / predict CJPE models. Model selection on the official dev split (macro-F1).

Each run writes to out_dir/run_name/: config.json, log.csv, best.pt (best dev macro-F1), dev_probs.json,
results.json (dev macro-F1 / P / R / accuracy, optional tuned threshold, trainable params, memory, time).
The `label` field of test.json is never read.
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

from ..amp import autocast as amp_autocast, describe as amp_describe, grad_scaler
from ..ner.model import apply_method, enable_grad_checkpointing, param_report
from ..trainer import load_trained
from . import data as cd
from .model import DocClassifier


@dataclass
class CJPEConfig:
    phase1_run: str = ""
    phase1_ckpt: str = "best"
    method: str = "full"                 # full | linear_probe | partial | <adapters/NAME.py> (lorma, lora, ...)
    method_kwargs: dict = field(default_factory=dict)
    data_dir: str = "/content/p4"        # train.json, dev.json, test.json
    out_dir: str = "/content/drive/MyDrive/cs781/cjpe_runs"
    run_name: str = ""
    # long-document representation
    strategy: str = "last"               # last | first_last | spread
    chunk_len: int = 512                 # tokens per chunk (<= the Phase-1 model's context)
    max_chunks: int = 4                  # chunks per document (1 = single window)
    token_pooling: str = "mean"          # mean | last   (token states -> chunk vector)
    chunk_agg: str = "mean"              # mean | max | attention   (chunk vectors -> document vector)
    normalize_text: bool = True          # collapse whitespace / line breaks
    drop_outcome: bool = False           # remove explicit-outcome sentences near the end (train 17.6% vs dev/test 8.2%)
    n_train: int = 0                     # stratified training subset (0 = all docs); the full train set is large
    # optimisation
    batch_size: int = 4                  # documents per forward pass
    grad_accum: int = 1
    grad_checkpoint: bool = False
    epochs: int = 3
    lr: float = 5e-5                     # backbone / adapter params
    head_lr: float = 1e-3                # head (+ chunk attention) params
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    class_weighting: str = "none"        # none | balanced
    tune_threshold: bool = False         # pick the P(label=1) threshold that maximises dev macro-F1 (used for test)
    eval_every: int = 0                  # also evaluate dev every N optimizer steps (0 = only at epoch ends)
    kd_teacher: str = ""                 # CJPE run dir of a trained teacher (same strategy/chunking) -> distillation
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0
    stop_at: str = ""                    # e.g. "2026-10-21T20:00+05:30": stop there (final dev eval, best.pt kept)
    seed: int = 1337

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"{os.path.basename(self.phase1_run.rstrip('/'))}__{self.method}__{self.strategy}{self.max_chunks}x{self.chunk_len}"


def binary_scores(gold, pred):
    out = {}
    f1s, ps, rs = [], [], []
    for c in (0, 1):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * P * R / (P + R) if P + R else 0.0); ps.append(P); rs.append(R)
    out.update(macro_f1=sum(f1s) / 2, macro_p=sum(ps) / 2, macro_r=sum(rs) / 2, f1_class0=f1s[0], f1_class1=f1s[1],
               accuracy=sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1))
    return out


def best_threshold(gold, prob1):
    cands = np.unique(np.round(np.asarray(prob1), 3))
    best = (0.5, binary_scores(gold, [int(p >= 0.5) for p in prob1])["macro_f1"])
    for t in cands:
        f = binary_scores(gold, [int(p >= t) for p in prob1])["macro_f1"]
        if f > best[1]:
            best = (float(t), f)
    return best


def _freeze_unused(clf, device):
    clf.train()
    ids = torch.randint(3, 1000, (1, 2, 16), device=device)
    clf.zero_grad(set_to_none=True)
    clf(ids, torch.ones_like(ids, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool, device=device)).float().sum().backward()
    for name, p in clf.named_parameters():
        if p.requires_grad and p.grad is None and ".experts." not in name:
            p.requires_grad = False
    clf.zero_grad(set_to_none=True)


def build_classifier(cfg, device):
    backbone, tok, p1cfg, mtype = load_trained(cfg.phase1_run, cfg.phase1_ckpt, device=device)
    if cfg.chunk_len > p1cfg.seq_len:
        raise ValueError(f"chunk_len {cfg.chunk_len} exceeds the Phase-1 context {p1cfg.seq_len}")
    clf = DocClassifier(backbone.float(), 2, cfg.token_pooling, cfg.chunk_agg, dropout=cfg.dropout).to(device)
    apply_method(clf, cfg.method, **cfg.method_kwargs)
    clf.to(device)
    _freeze_unused(clf, device)
    if cfg.grad_checkpoint:
        print(f"gradient checkpointing on {enable_grad_checkpointing(clf.backbone)} blocks")
    return clf, tok, mtype, param_report(clf)


def load_cjpe_run(run_dir, device="cuda"):
    saved = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = CJPEConfig(**{k: v for k, v in saved.items() if k in CJPEConfig.__dataclass_fields__})
    clf, tok, mtype, _ = build_classifier(cfg, device)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    clf.load_state_dict(ck["state"])
    return clf.eval(), tok, cfg, ck.get("threshold", 0.5)


def _batches(examples, bs, chunk_len, pad_id, shuffle, rng):
    ex = list(examples)
    if shuffle:
        rng.shuffle(ex)
    for i in range(0, len(ex), bs):
        b = ex[i:i + bs]
        C = max(len(e["chunks"]) for e in b)
        L = max(len(c) for e in b for c in e["chunks"])
        ids = np.full((len(b), C, L), pad_id, dtype=np.int64)
        tmask = np.zeros((len(b), C, L), dtype=bool)
        cmask = np.zeros((len(b), C), dtype=bool)
        for k, e in enumerate(b):
            for j, c in enumerate(e["chunks"]):
                ids[k, j, :len(c)] = c
                tmask[k, j, :len(c)] = True
                cmask[k, j] = True
        tmask[~cmask] = False
        tmask[:, :, 0] |= ~cmask                 # padded chunks get one dummy token (masked out by chunk_mask)
        yield b, torch.from_numpy(ids), torch.from_numpy(tmask), torch.from_numpy(cmask)


@torch.no_grad()
def predict_probs(clf, examples, bs, chunk_len, device, pad_id):
    clf.eval()
    amp = amp_autocast(device)
    out = {}
    for b, ids, tm, cm in _batches(examples, bs, chunk_len, pad_id, False, None):
        with amp:
            p = torch.softmax(clf(ids.to(device), tm.to(device), cm.to(device)).float(), -1)[:, 1].cpu().tolist()
        for e, pr in zip(b, p):
            out[e["id"]] = pr
    return out


def _train_cjpe(cfg: CJPEConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    t0 = time.time()
    print(f"[{time.time() - t0:5.0f}s] loading Phase-1 model + applying '{cfg.method}' ...", flush=True)
    clf, tok, mtype, report = build_classifier(cfg, device)
    pad_id = tok.token_to_id("<|pad|>")
    print(f"[{time.time() - t0:5.0f}s] loading + selecting chunks ({cfg.strategy}, {cfg.max_chunks} x {cfg.chunk_len} tokens) ...", flush=True)
    train_docs = cd.subsample(cd.load_split(os.path.join(cfg.data_dir, "train.json")), cfg.n_train, cfg.seed)
    dev_docs = cd.load_split(os.path.join(cfg.data_dir, "dev.json"))
    train = cd.prepare(train_docs, tok, cfg.strategy, cfg.chunk_len, cfg.max_chunks, cfg.normalize_text, cfg.drop_outcome)
    dev = cd.prepare(dev_docs, tok, cfg.strategy, cfg.chunk_len, cfg.max_chunks, cfg.normalize_text, cfg.drop_outcome)
    del train_docs
    dev_gold = [d["label"] for d in dev]
    n1 = sum(d["label"] for d in train)
    print(f"[{time.time() - t0:5.0f}s] {cfg.run_name}: {report['trainable_params'] / 1e6:.2f}M trainable of "
          f"{report['total_params'] / 1e6:.1f}M ({report['trainable_pct']:.2f}%) | train {len(train)} docs "
          f"(label 1: {n1}, label 0: {len(train) - n1}) | dev {len(dev)} | precision: {amp_describe(device)}", flush=True)

    weight = None
    if cfg.class_weighting == "balanced":
        c = np.array([len(train) - n1, n1], dtype=np.float64) + 1
        weight = torch.tensor(c.sum() / (2 * c), dtype=torch.float32, device=device)
    teacher = None
    if cfg.kd_teacher:
        teacher, _, t_cfg, _ = load_cjpe_run(cfg.kd_teacher, device)
        assert (t_cfg.strategy, t_cfg.chunk_len, t_cfg.max_chunks) == (cfg.strategy, cfg.chunk_len, cfg.max_chunks), \
            "KD here needs the teacher to see the same chunks (strategy / chunk_len / max_chunks)"
        for p in teacher.parameters():
            p.requires_grad = False

    bb_ids = {id(p) for p in clf.backbone.parameters()}
    body = [p for p in clf.parameters() if p.requires_grad and id(p) in bb_ids]
    head = [p for p in clf.parameters() if p.requires_grad and id(p) not in bb_ids]   # head + chunk attention
    opt = torch.optim.AdamW([dict(params=body, lr=cfg.lr), dict(params=head, lr=cfg.head_lr)], weight_decay=cfg.weight_decay)
    accum = max(1, cfg.grad_accum)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size)
    total = math.ceil(steps_per_epoch / accum) * cfg.epochs
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp, scaler = amp_autocast(device), grad_scaler(device)
    deadline = None
    if cfg.stop_at:
        from datetime import datetime
        deadline = datetime.fromisoformat(cfg.stop_at).timestamp()
        print(f"will stop at {cfg.stop_at} ({(deadline - time.time()) / 60:.0f} min from now)")

    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "opt_step", "train_loss", "dev_macro_f1", "dev_macro_p", "dev_macro_r", "dev_accuracy", "seconds"])
    state = dict(best=-1.0, best_scores=None, opt_steps=0)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def evaluate(ep, loss_avg, te):
        probs = predict_probs(clf, dev, cfg.batch_size * 2, cfg.chunk_len, device, pad_id)
        pred = [int(probs[d["id"]] >= 0.5) for d in dev]
        s = binary_scores(dev_gold, pred)
        lw.writerow([ep, state["opt_steps"], round(loss_avg, 5), round(s["macro_f1"], 5), round(s["macro_p"], 5),
                     round(s["macro_r"], 5), round(s["accuracy"], 5), round(time.time() - te, 1)]); log.flush()
        print(f"\n  [dev] epoch {ep} step {state['opt_steps']}: macro-F1 {s['macro_f1']:.4f} (P {s['macro_p']:.3f}, "
              f"R {s['macro_r']:.3f}, acc {s['accuracy']:.4f})", flush=True)
        if s["macro_f1"] > state["best"]:
            state["best"], state["best_scores"] = s["macro_f1"], s
            torch.save(dict(state=clf.state_dict(), epoch=ep, scores=s, threshold=0.5), os.path.join(run_dir, "best.pt"))
            json.dump(probs, open(os.path.join(run_dir, "dev_probs.json"), "w"))
        clf.train()

    def optimizer_step():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in clf.parameters() if p.requires_grad], cfg.grad_clip)
        scaler.step(opt); scaler.update(); sched.step()
        opt.zero_grad(set_to_none=True)
        state["opt_steps"] += 1

    rng, train_time, stopped = random.Random(cfg.seed), 0.0, False
    opt.zero_grad(set_to_none=True)
    for ep in range(1, cfg.epochs + 1):
        clf.train()
        te, tot, n = time.time(), 0.0, 0
        for b, ids, tm, cm in _batches(train, cfg.batch_size, cfg.chunk_len, pad_id, True, rng):
            if deadline and time.time() >= deadline:
                stopped = True
                print(f"\nstop_at reached in epoch {ep} after {n} batches: final dev evaluation, then stopping")
                break
            if n % 20 == 0:
                el = time.time() - te
                eta = f"~{el / n * (steps_per_epoch - n) / 60:.1f} min left in epoch" if n else "starting"
                print(f"  epoch {ep} batch {n}/{steps_per_epoch} | loss {tot / max(n, 1):.4f} | {el / 60:.1f} min, {eta}",
                      end="\r", flush=True)
            lab = torch.tensor([e["label"] for e in b], device=device)
            ids, tm, cm = ids.to(device), tm.to(device), cm.to(device)
            with amp:
                logits = clf(ids, tm, cm).float()
            loss = F.cross_entropy(logits, lab, weight=weight)
            if teacher is not None:
                with torch.no_grad(), amp:
                    t_logits = teacher(ids, tm, cm).float()
                T = cfg.kd_temperature
                loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * T * T * F.kl_div(
                    F.log_softmax(logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean")
            scaler.scale(loss / accum).backward()
            tot += loss.item(); n += 1
            if n % accum == 0:
                optimizer_step()
                if cfg.eval_every and state["opt_steps"] % cfg.eval_every == 0:
                    evaluate(ep, tot / n, te)
        if n % accum:
            optimizer_step()
        train_time += time.time() - te
        evaluate(ep, tot / max(n, 1), te)
        if stopped:
            break
    log.close()

    thr = 0.5
    if cfg.tune_threshold and state["best_scores"]:
        probs = json.load(open(os.path.join(run_dir, "dev_probs.json")))
        thr, f = best_threshold(dev_gold, [probs[d["id"]] for d in dev])
        ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
        ck["threshold"] = thr
        torch.save(ck, os.path.join(run_dir, "best.pt"))
        print(f"tuned threshold on dev: {thr:.3f} -> dev macro-F1 {f:.4f} (0.5 gave {state['best']:.4f}; "
              f"optimistic, since it is chosen on the same dev set)")
    results = dict(run=cfg.run_name, method=cfg.method, phase1_run=os.path.basename(cfg.phase1_run.rstrip("/")),
                   strategy=cfg.strategy, drop_outcome=cfg.drop_outcome, chunk_len=cfg.chunk_len, max_chunks=cfg.max_chunks, chunk_agg=cfg.chunk_agg,
                   token_pooling=cfg.token_pooling, n_train=len(train),
                   kd_teacher=os.path.basename(cfg.kd_teacher.rstrip("/")) if cfg.kd_teacher else "",
                   best_dev_macro_f1=state["best"], best_dev_scores=state["best_scores"], threshold=thr,
                   stopped_early_at=cfg.stop_at if stopped else None, train_seconds=round(train_time, 1),
                   total_seconds=round(time.time() - t0, 1),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None, **report)
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    return results


def train_cjpe(cfg: CJPEConfig):
    from ..rr.trainer import free_gpu_after_failure
    try:
        return _train_cjpe(cfg)
    except BaseException as e:
        free_gpu_after_failure(e)
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print("Out of GPU memory: lower batch_size / max_chunks / chunk_len, raise grad_accum, "
                  "set grad_checkpoint=True, or use a parameter-efficient method.")
        raise


def predict_test(run_dir, out_dir=None, device="cuda", bs=8):
    """test.json -> prediction.json [{"id": ..., "pred": 0|1}, ...] + submission.zip. Test labels are not read."""
    from ..submit_json import write_json_submission
    clf, tok, cfg, thr = load_cjpe_run(run_dir, device)
    docs = cd.load_split(os.path.join(cfg.data_dir, "test.json"))
    ex = cd.prepare([dict(d, label=None) for d in docs], tok, cfg.strategy, cfg.chunk_len, cfg.max_chunks, cfg.normalize_text, cfg.drop_outcome)
    t0 = time.time()
    probs = predict_probs(clf, ex, bs, cfg.chunk_len, device, tok.token_to_id("<|pad|>"))
    preds = [{"id": d["id"], "pred": int(probs[d["id"]] >= thr)} for d in docs]
    zip_path = write_json_submission(preds, expected_ids={d["id"] for d in docs}, out_dir=out_dir or os.path.join(run_dir, "submission"))
    ones = sum(p["pred"] for p in preds)
    print(f"{len(preds)} test cases ({time.time() - t0:.0f}s): {ones} predicted 1, {len(preds) - ones} predicted 0 (threshold {thr:.3f})")
    return zip_path, preds
