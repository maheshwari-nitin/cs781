"""Train / evaluate / predict Rhetorical Role models. Model selection on the official dev sets (CL_dev + IT_dev).

Each run writes to out_dir/run_name/: config.json, log.csv, best.pt (best dev macro-F1), dev_pred.json,
results.json (overall + per-domain macro-F1, per-label F1, trainable params, memory, time).
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

from ..ner.model import apply_method, param_report
from ..amp import autocast as amp_autocast, describe as amp_describe, grad_scaler
from ..trainer import load_trained
from . import data as rd
from .model import SentenceClassifier


@dataclass
class RRConfig:
    phase1_run: str = ""
    phase1_ckpt: str = "best"
    method: str = "full"                 # full | linear_probe | partial | <adapters/NAME.py> (lorma, lora, ...)
    method_kwargs: dict = field(default_factory=dict)
    data_dir: str = "/content/p3"        # {CL,IT}_{train,dev,test}.json + label_vocab.json
    out_dir: str = "/content/drive/MyDrive/cs781/rr_runs"
    run_name: str = ""
    domains: list = field(default_factory=lambda: ["CL", "IT"])   # which domains to train on
    context: str = "document"            # document (sentences read with neighbours) | sentence (no context)
    pooling: str = "mean"                # mean | last
    window: int = 1024
    stride: int = 768
    max_sent_tokens: int = 256
    train_on_dev: bool = False           # final model: add dev docs to training (no model selection then)
    batch_size: int = 8                  # windows per forward pass (physical batch)
    grad_accum: int = 1                  # optimizer step every N batches: effective batch = batch_size * grad_accum
    grad_checkpoint: bool = False        # recompute block activations in backward: far less GPU memory (T4), ~30% slower
    stop_at: str = ""                    # wall-clock deadline with timezone, e.g. "2026-10-16T08:30+05:30": training stops
                                         # there (final dev eval + best.pt kept) and train_rr returns normally
    epochs: int = 10
    lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    class_weighting: str = "none"        # none | inv_sqrt | inv   (macro-F1 rewards rare roles)
    extra_train: list = field(default_factory=list)  # extra train-format JSON files, e.g. synthetic/CL_synth.json
                                         # (domain taken from the file name prefix CL_/IT_); declare their origin
    # augmentation from the official training docs only (lm_lib/rr/augment.py); dev/test are never touched
    augment_docs: int = 0                # synthetic segment-swap documents per domain (0 = off)
    augment_rare_boost: float = 3.0      # template weight for training docs that contain a rare role
    oversample_rare: int = 0             # extra copies of training windows containing a rare-role sentence
    rare_threshold: int = 500            # roles with fewer training sentences than this count as rare
    kd_teacher: str = ""                 # RR run dir of a trained teacher -> knowledge distillation
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0
    seed: int = 1337

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"{os.path.basename(self.phase1_run.rstrip('/'))}__{self.method}__{self.context}"


def macro_f1(gold, pred, labels):
    """sklearn-style macro-F1 over the labels that occur in gold or predictions."""
    present = sorted(set(gold) | set(pred))
    per = {}
    for l in present:
        tp = sum(1 for g, p in zip(gold, pred) if g == l and p == l)
        fp = sum(1 for g, p in zip(gold, pred) if g != l and p == l)
        fn = sum(1 for g, p in zip(gold, pred) if g == l and p != l)
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        per[labels[l]] = dict(f1=2 * P * R / (P + R) if P + R else 0.0, p=P, r=R, support=tp + fn)
    k = max(len(per), 1)
    return dict(macro_f1=sum(v["f1"] for v in per.values()) / k, macro_p=sum(v["p"] for v in per.values()) / k,
                macro_r=sum(v["r"] for v in per.values()) / k,
                accuracy=sum(g == p for g, p in zip(gold, pred)) / max(len(gold), 1), per_label=per)


def score_docs(preds, ds, labels):
    """preds: {doc_id: [label ids]} -> overall + per-domain scores."""
    out = {}
    for dom in [None] + sorted({d["domain"] for d in ds.docs.values()}):
        g, p = [], []
        for d, doc in ds.docs.items():
            if dom is None or doc["domain"] == dom:
                g += doc["labels"]; p += preds[d]
        out["all" if dom is None else dom] = macro_f1(g, p, labels)
    return out


def _freeze_unused(clf, device):
    clf.train()
    ids = torch.randint(3, 1000, (1, 32), device=device)
    segs = torch.tensor([[0, 0, 16], [0, 16, 32]], device=device)
    clf.zero_grad(set_to_none=True)
    clf(ids, segs).float().sum().backward()
    for name, p in clf.named_parameters():
        if p.requires_grad and p.grad is None and ".experts." not in name:
            p.requires_grad = False
    clf.zero_grad(set_to_none=True)


def build_classifier(cfg, n_labels, device):
    backbone, tok, p1cfg, mtype = load_trained(cfg.phase1_run, cfg.phase1_ckpt, device=device)
    clf = SentenceClassifier(backbone.float(), n_labels, cfg.pooling, dropout=cfg.dropout).to(device)
    apply_method(clf, cfg.method, **cfg.method_kwargs)
    clf.to(device)
    _freeze_unused(clf, device)
    if cfg.grad_checkpoint:
        from ..ner.model import enable_grad_checkpointing
        print(f"gradient checkpointing on {enable_grad_checkpointing(clf.backbone)} blocks")
    return clf, tok, mtype, param_report(clf)


def load_rr_run(run_dir, device="cuda"):
    saved = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = RRConfig(**{k: v for k, v in saved.items() if k in RRConfig.__dataclass_fields__})
    labels, label2id = rd.load_labels(os.path.join(cfg.data_dir, "label_vocab.json"))
    clf, tok, mtype, _ = build_classifier(cfg, len(labels), device)
    clf.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)["state"])
    return clf.eval(), tok, mtype, cfg, (labels, label2id)


def _batches(ds, examples, bs, pad_id, shuffle, rng):
    ex = list(examples)
    if shuffle:
        rng.shuffle(ex)
    chunks = [ex[i:i + bs * 8] for i in range(0, len(ex), bs * 8)]
    groups = []
    for ch in chunks:
        ch.sort(key=lambda e: e[2] - e[1])                 # similar lengths -> little right-padding
        groups += [ch[i:i + bs] for i in range(0, len(ch), bs)]
    if shuffle:
        rng.shuffle(groups)
    for b in groups:
        L = max(z - a for _, a, z, _ in b)
        ids = np.full((len(b), L), pad_id, dtype=np.int64)
        segs, lab, who = [], [], []
        for k, (d, a, z, sents) in enumerate(b):
            doc = ds.docs[d]
            ids[k, :z - a] = doc["ids"][a:z]
            for i, s, e in sents:
                segs.append((k, s, e))
                lab.append(doc["labels"][i] if doc["labels"] else -100)
                who.append((d, i))
        if segs:
            yield torch.from_numpy(ids), torch.tensor(segs), torch.tensor(lab), who


@torch.no_grad()
def predict(clf, ds, bs, device, pad_id, return_logits=False):
    clf.eval()
    amp = amp_autocast(device)
    out = {d: [None] * doc["n"] for d, doc in ds.docs.items()}
    logit_out = {d: [None] * doc["n"] for d, doc in ds.docs.items()}
    for ids, segs, _, who in _batches(ds, ds.pred_examples, bs, pad_id, False, None):
        with amp:
            logits = clf(ids.to(device), segs.to(device)).float().cpu()
        for (d, i), lg in zip(who, logits):
            out[d][i] = int(lg.argmax())
            logit_out[d][i] = lg.numpy()
    for d, v in out.items():
        assert all(x is not None for x in v), f"doc {d}: some sentences got no prediction"
    return (out, logit_out) if return_logits else out


def _train_rr(cfg: RRConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(asdict(cfg), open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    labels, label2id = rd.load_labels(os.path.join(cfg.data_dir, "label_vocab.json"))

    t0 = time.time()
    print(f"[{time.time() - t0:5.0f}s] loading Phase-1 model {cfg.phase1_run} ({cfg.phase1_ckpt}) + applying '{cfg.method}' ...", flush=True)
    clf, tok, mtype, report = build_classifier(cfg, len(labels), device)
    causal = mtype == "decoder"
    pad_id = tok.token_to_id("<|pad|>")
    train_docs = rd.load_docs(cfg.data_dir, "train", cfg.domains)
    dev_docs = rd.load_docs(cfg.data_dir, "dev", rd.DOMAINS)
    if cfg.train_on_dev:
        train_docs += [d for d in dev_docs if d["domain"] in cfg.domains]
    n_extra = 0
    for path in cfg.extra_train:
        dom = os.path.basename(path).split("_")[0]
        if dom not in cfg.domains:
            continue
        for x in json.load(open(path, encoding="utf-8")):
            assert len(x["text"]) == len(x["labels"]) and all(l in label2id for l in x["labels"]), f"bad doc in {path}"
            train_docs.append(dict(id=str(x["id"]), domain=dom, text=list(x["text"]), labels=list(x["labels"])))
            n_extra += 1
    if n_extra:
        print(f"extra training docs: {n_extra} from {[os.path.basename(p) for p in cfg.extra_train]}")
    from . import augment as aug
    rare = aug.rare_roles(train_docs, cfg.rare_threshold)
    synth = aug.segment_swap_docs(train_docs, cfg.augment_docs, rare, cfg.augment_rare_boost, cfg.seed) if cfg.augment_docs else []
    aug_summary = aug.summary(train_docs, synth) if synth else None
    if synth:
        print(f"augmentation: {len(synth)} synthetic segment-swap docs; rare roles {sorted(rare)}")
        for l, v in aug_summary.items():
            print(f"  {l:24s} real {v['real']:6d}  + synthetic {v['synthetic']:6d}")
    print(f"[{time.time() - t0:5.0f}s] tokenising {len(train_docs) + len(synth)} train + {len(dev_docs)} dev docs ...", flush=True)
    mk = lambda docs: rd.RRDataset(docs, tok, label2id, cfg.window, cfg.stride, cfg.context, causal, cfg.max_sent_tokens)
    train_ds = mk(train_docs + synth)
    if cfg.oversample_rare:
        before = len(train_ds.train_examples)
        train_ds.train_examples = aug.oversample_rare_windows(train_ds.train_examples, train_ds.docs,
                                                              {label2id[l] for l in rare}, cfg.oversample_rare)
        print(f"oversampling: {before} -> {len(train_ds.train_examples)} training examples")
    dev_ds = None if cfg.train_on_dev else mk(dev_docs)
    print(f"[{time.time() - t0:5.0f}s] data ready", flush=True)
    print(f"{cfg.run_name}: {report['trainable_params'] / 1e6:.2f}M trainable of {report['total_params'] / 1e6:.1f}M "
          f"({report['trainable_pct']:.2f}%) | {'causal' if causal else 'bidirectional'} features, {cfg.pooling} pooling, "
          f"{cfg.context} context | train {len(train_docs)} docs / {len(train_ds.train_examples)} examples")

    weight = None
    if cfg.class_weighting in ("inv_sqrt", "inv"):
        c = train_ds.label_counts(len(labels)).astype(np.float64) + 1.0
        w = (c.sum() / c) ** (0.5 if cfg.class_weighting == "inv_sqrt" else 1.0)
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    teacher = None
    if cfg.kd_teacher:
        teacher, t_tok, t_mtype, t_cfg, _ = load_rr_run(cfg.kd_teacher, device)
        assert t_cfg.context == cfg.context and t_cfg.window == cfg.window and t_cfg.stride == cfg.stride, \
            "KD here needs the teacher to see the same windows (same context/window/stride)"
        for p in teacher.parameters():
            p.requires_grad = False

    head_ids = {id(p) for p in clf.head.parameters()}
    body = [p for p in clf.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([dict(params=body, lr=cfg.lr), dict(params=list(clf.head.parameters()), lr=cfg.head_lr)],
                            weight_decay=cfg.weight_decay)
    accum = max(1, cfg.grad_accum)
    total = math.ceil(math.ceil(len(train_ds.train_examples) / cfg.batch_size) / accum) * cfg.epochs   # optimizer steps
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp = amp_autocast(device)

    scaler = grad_scaler(device)
    print(f"precision: {amp_describe(device)}")
    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "dev_macro_f1", "dev_CL_macro_f1", "dev_IT_macro_f1", "dev_accuracy", "seconds"])
    best, best_scores, rng, train_time = -1.0, None, random.Random(cfg.seed), 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps_per_epoch = math.ceil(len(train_ds.train_examples) / cfg.batch_size)

    def optimizer_step():
        scaler.unscale_(opt)                           # no-op unless fp16 (then clip the true gradients)
        torch.nn.utils.clip_grad_norm_([p for p in clf.parameters() if p.requires_grad], cfg.grad_clip)
        scaler.step(opt); scaler.update(); sched.step()
        opt.zero_grad(set_to_none=True)

    deadline = None
    if cfg.stop_at:
        from datetime import datetime
        deadline = datetime.fromisoformat(cfg.stop_at).timestamp()
        print(f"will stop at {cfg.stop_at} ({(deadline - time.time()) / 60:.0f} min from now)")
    stopped = False
    opt.zero_grad(set_to_none=True)
    for ep in range(1, cfg.epochs + 1):
        clf.train()
        te, tot, n = time.time(), 0.0, 0
        for ids, segs, lab, _ in _batches(train_ds, train_ds.train_examples, cfg.batch_size, pad_id, True, rng):
            if deadline and time.time() >= deadline:
                stopped = True
                print()
                print(f"stop_at reached during epoch {ep} (after {n} steps): final dev evaluation, then stopping")
                break
            if n % 20 == 0:
                el = time.time() - te
                eta = f"~{el / n * (steps_per_epoch - n):.0f}s left in epoch" if n else "starting"
                print(f"  epoch {ep} step {n}/{steps_per_epoch} | loss {tot / max(n, 1):.4f} | {el:.0f}s elapsed, {eta}",
                      end="\r", flush=True)
            ids, segs, lab = ids.to(device), segs.to(device), lab.to(device)
            with amp:
                logits = clf(ids, segs).float()
            loss = F.cross_entropy(logits, lab, weight=weight)
            if teacher is not None:
                with torch.no_grad(), amp:
                    t_logits = teacher(ids, segs).float()
                T = cfg.kd_temperature
                kl = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(t_logits / T, -1), reduction="batchmean")
                loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * T * T * kl
            scaler.scale(loss / accum).backward()
            tot += loss.item(); n += 1
            if n % accum == 0:
                optimizer_step()
        if n % accum:                                  # flush a partial accumulation at the end of the epoch
            optimizer_step()
        train_time += time.time() - te
        row = [ep, round(tot / max(n, 1), 5)]
        if dev_ds is not None:
            print(f"  epoch {ep}: {n} steps in {time.time() - te:.0f}s, evaluating dev ...".ljust(90), end="\r", flush=True)
            preds = predict(clf, dev_ds, cfg.batch_size * 2, device, pad_id)
            s = score_docs(preds, dev_ds, labels)
            row += [round(s["all"]["macro_f1"], 5), round(s.get("CL", {}).get("macro_f1", 0), 5),
                    round(s.get("IT", {}).get("macro_f1", 0), 5), round(s["all"]["accuracy"], 5)]
            print(f"epoch {ep}: loss {row[1]:.4f} | dev macro-F1 {row[2]:.4f} (CL {row[3]:.4f}, IT {row[4]:.4f}) | "
                  f"acc {row[5]:.4f} | {time.time() - te:.0f}s")
            if s["all"]["macro_f1"] > best:
                best, best_scores = s["all"]["macro_f1"], s
                torch.save(dict(state=clf.state_dict(), epoch=ep, scores=s), os.path.join(run_dir, "best.pt"))
                json.dump({d: [labels[i] for i in v] for d, v in preds.items()}, open(os.path.join(run_dir, "dev_pred.json"), "w"))
        else:
            row += ["", "", "", ""]
            print(f"epoch {ep}: loss {row[1]:.4f} | {time.time() - te:.0f}s")
        lw.writerow(row + [round(time.time() - te, 1)]); log.flush()
        if stopped:
            break
    log.close()
    if dev_ds is None:
        torch.save(dict(state=clf.state_dict(), epoch=cfg.epochs, scores=None), os.path.join(run_dir, "best.pt"))
    results = dict(run=cfg.run_name, method=cfg.method, context=cfg.context, pooling=cfg.pooling, stopped_early_at=cfg.stop_at if stopped else None,
                   phase1_run=os.path.basename(cfg.phase1_run.rstrip("/")),
                   kd_teacher=os.path.basename(cfg.kd_teacher.rstrip("/")) if cfg.kd_teacher else "",
                   best_dev_macro_f1=best if dev_ds is not None else None,
                   dev_CL_macro_f1=best_scores["CL"]["macro_f1"] if best_scores else None,
                   dev_IT_macro_f1=best_scores["IT"]["macro_f1"] if best_scores else None,
                   dev_per_label_f1={k: round(v["f1"], 4) for k, v in best_scores["all"]["per_label"].items()} if best_scores else None,
                   augment_docs=cfg.augment_docs, oversample_rare=cfg.oversample_rare, augmentation=aug_summary,
                   extra_train=[os.path.basename(p) for p in cfg.extra_train], extra_train_docs=n_extra,
                   train_seconds=round(train_time, 1), total_seconds=round(time.time() - t0, 1),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None, **report)
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    if best_scores:
        print("\nbest dev epoch, per label:")
        for l, v in sorted(best_scores["all"]["per_label"].items(), key=lambda x: -x[1]["f1"]):
            print(f"  {l:24s} F1 {v['f1']:.3f}  P {v['p']:.3f}  R {v['r']:.3f}  (support {v['support']})")
    return results


def free_gpu_after_failure(exc):
    """A failed run's traceback keeps every frame (model, optimizer, activations) alive, so GPU memory is never
    released and the next attempt runs out of memory while loading. Drop those frames and empty the cache."""
    import gc
    import traceback
    traceback.clear_frames(exc.__traceback__)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[cleanup after error] GPU memory in use now: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def train_rr(cfg: RRConfig):
    try:
        return _train_rr(cfg)
    except BaseException as e:            # incl. KeyboardInterrupt and CUDA OOM
        free_gpu_after_failure(e)
        if isinstance(e, torch.cuda.OutOfMemoryError):
            print("Out of GPU memory: lower batch_size (and raise grad_accum to keep the effective batch), "
                  "set grad_checkpoint=True, or use a parameter-efficient method.")
        raise


def predict_test(run_dir, out_dir=None, device="cuda", bs=16):
    """CL_test + IT_test -> prediction.json {doc_id: [label per sentence]} + submission.zip, with length checks."""
    from ..submit_json import write_json_submission
    clf, tok, mtype, cfg, (labels, label2id) = load_rr_run(run_dir, device)
    docs = rd.load_docs(cfg.data_dir, "test", rd.DOMAINS)
    ds = rd.RRDataset(docs, tok, label2id, cfg.window, cfg.stride, cfg.context, mtype == "decoder", cfg.max_sent_tokens)
    t0 = time.time()
    preds = predict(clf, ds, bs, device, tok.token_to_id("<|pad|>"))
    out = {d: [labels[i] for i in preds[d]] for d in preds}
    for doc in docs:
        assert len(out[doc["id"]]) == len(doc["text"]), f"{doc['id']}: {len(out[doc['id']])} preds vs {len(doc['text'])} sentences"
    zip_path = write_json_submission(out, expected_ids={d["id"] for d in docs}, out_dir=out_dir or os.path.join(run_dir, "submission"))
    print(f"{sum(len(v) for v in out.values())} sentences over {len(out)} test docs ({time.time() - t0:.1f}s)")
    return zip_path, out
