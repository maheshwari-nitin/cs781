"""Train / evaluate / predict a token classifier on top of a Phase-1 model.

Every run writes to out_dir/run_name/:
  config.json     the NERConfig
  log.csv         per-epoch train loss, val strict macro-F1 / P / R, time
  best.pt         classifier state_dict at the best val macro-F1 (or the last epoch if there is no val split)
  val_pred.json   val predictions of the best epoch;  results.json  scores + efficiency numbers for the report
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

from ..trainer import load_trained
from . import data as nd
from .metrics import format_scores, strict_scores
from .model import TokenClassifier, apply_method, param_report


@dataclass
class NERConfig:
    phase1_run: str = ""                 # Phase-1 run dir (e.g. .../runs/A_qwen_dense_full)
    phase1_ckpt: str = "best"            # best | final | step_XXXXXX
    method: str = "full"                 # full | linear_probe | partial | <your adapters/NAME.py>
    method_kwargs: dict = field(default_factory=dict)
    data_dir: str = "/content/p2"        # fold1/2-text/gold.json, fold3-text.json, labels.txt
    out_dir: str = "/content/drive/MyDrive/cs781/ner_runs"
    run_name: str = ""
    window: int = 1024
    stride: int = 768
    n_val_docs: int = 10                 # held out from the 70 training docs; 0 => train on all (no model selection)
    batch_size: int = 8
    epochs: int = 10
    lr: float = 5e-5                     # backbone / adapter params
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    dropout: float = 0.1
    class_weighting: str = "none"        # none | inv_sqrt  (counter the ~95% "O" tokens)
    kd_teacher: str = ""                 # NER run dir of a trained teacher => knowledge distillation
    kd_alpha: float = 0.5                # loss = (1 - alpha) * CE + alpha * T^2 * KL(teacher_T || student_T)
    kd_temperature: float = 2.0
    decode: str = "argmax"               # argmax | viterbi (constrained BIO)   -- used for val selection + test
    window_agg: str = "owner"            # owner | mean   (how overlapping windows are combined)
    seed: int = 1337

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"{os.path.basename(self.phase1_run.rstrip('/'))}__{self.method}"


def build_classifier(cfg: NERConfig, n_tags, device):
    backbone, tok, p1cfg, mtype = load_trained(cfg.phase1_run, cfg.phase1_ckpt, device=device)
    clf = TokenClassifier(backbone.float(), n_tags, dropout=cfg.dropout).to(device)
    apply_method(clf, cfg.method, **cfg.method_kwargs)
    clf.to(device)                      # adapter modules created by apply() land on the right device
    _freeze_unused(clf, device)
    report = param_report(clf)
    return clf, tok, mtype, report


def _freeze_unused(clf, device):
    """Params that never get a gradient (e.g. the decoder of an encoder-decoder when only its encoder makes
    the features) are frozen so trainable-parameter counts are honest. MoE experts are left alone (their
    usage depends on routing)."""
    clf.train()
    ids = torch.randint(3, 1000, (1, 32), device=device)
    clf.zero_grad(set_to_none=True)
    clf(ids).float().sum().backward()
    for name, p in clf.named_parameters():
        if p.requires_grad and p.grad is None and ".experts." not in name:
            p.requires_grad = False
    clf.zero_grad(set_to_none=True)


def load_ner_run(run_dir, device="cuda"):
    saved = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = NERConfig(**{k: v for k, v in saved.items() if k in NERConfig.__dataclass_fields__})
    labels, tags, tag2id = nd.load_labels(os.path.join(cfg.data_dir, "labels.txt"))
    clf, tok, mtype, _ = build_classifier(cfg, len(tags), device)
    clf.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)["state"])
    return clf.eval(), tok, mtype, cfg, (labels, tags, tag2id)


def _batches(ds, examples, bs, pad_id, shuffle, rng):
    ex = list(examples)
    if shuffle:
        rng.shuffle(ex)
    # group similar lengths so right-padding is minimal (matters for bidirectional encoders, which have no mask)
    chunks = [ex[i:i + bs * 8] for i in range(0, len(ex), bs * 8)]
    out = []
    for ch in chunks:
        ch.sort(key=lambda e: e[2] - e[1])
        out += [ch[i:i + bs] for i in range(0, len(ch), bs)]
    if shuffle:
        rng.shuffle(out)
    for b in out:
        L = max(e[2] - e[1] for e in b)
        ids = np.full((len(b), L), pad_id, dtype=np.int64)
        lab = np.full((len(b), L), -100, dtype=np.int64)
        for i, (d, a, z) in enumerate(b):
            doc = ds.docs[d]
            ids[i, :z - a] = doc["ids"][a:z]
            if doc["labels"] is not None:
                lab[i, :z - a] = doc["labels"][a:z]
        yield b, torch.from_numpy(ids), torch.from_numpy(lab)


@torch.no_grad()
def window_logprobs(clf, ds, bs, device, pad_id=1):
    """Run every window once -> {(doc, window_start): log-probs [window_len, n_tags] (numpy float32)}."""
    clf.eval()
    out = {}
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
    for b, ids, _ in _batches(ds, [(d, a, z) for d, doc in ds.docs.items() for a, z in doc["windows"]], bs, pad_id,
                              False, None):
        with amp:
            logits = clf(ids.to(device)).float()
        lp = torch.log_softmax(logits, -1).cpu().numpy()
        for i, (d, a, z) in enumerate(b):
            out[(d, a)] = lp[i, :z - a]
    return out


def decode_docs(win_lp, ds, texts, tags, causal, decode="argmax", window_agg="owner"):
    """Per-token log-probs from the windows -> tag sequence -> char spans.
    window_agg: 'owner' = the window where the token has most context; 'mean' = average over all covering windows.
    decode:     'argmax' = independent per-token argmax; 'viterbi' = best *valid* BIO sequence."""
    preds = {}
    for d, doc in ds.docs.items():
        n, wins = len(doc["ids"]), doc["windows"]
        if n == 0:
            preds[d] = []
            continue
        lp = np.zeros((n, len(tags)), dtype=np.float32)
        if window_agg == "owner":
            owner = nd.token_owner(wins, n, causal)
            for w, (a, z) in enumerate(wins):
                sel = np.where(owner[a:z] == w)[0]
                lp[a + sel] = win_lp[(d, a)][sel]
        elif window_agg == "mean":
            cnt = np.zeros((n, 1), dtype=np.float32)
            for a, z in wins:
                lp[a:z] += win_lp[(d, a)]
                cnt[a:z] += 1
            lp /= cnt
        else:
            raise ValueError(window_agg)
        tag_ids = nd.viterbi_bio(lp, tags) if decode == "viterbi" else lp.argmax(-1)
        preds[d] = nd.bio_to_spans(tag_ids, doc["spans"], texts[d], tags)
    return preds


def predict_docs(clf, ds, texts, tags, causal, bs, device, pad_id=1, decode="argmax", window_agg="owner"):
    return decode_docs(window_logprobs(clf, ds, bs, device, pad_id), ds, texts, tags, causal, decode, window_agg)


def train_ner(cfg: NERConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    labels, tags, tag2id = nd.load_labels(os.path.join(cfg.data_dir, "labels.txt"))
    texts, golds = nd.load_folds(cfg.data_dir, [1, 2])
    if cfg.n_val_docs:
        (tr_t, tr_g), (va_t, va_g) = nd.split_docs(texts, golds, cfg.n_val_docs, cfg.seed)
    else:
        (tr_t, tr_g), (va_t, va_g) = (texts, golds), ({}, {})

    t0 = time.time()
    clf, tok, mtype, report = build_classifier(cfg, len(tags), device)
    causal = mtype == "decoder"
    print(f"{cfg.run_name}: {report['trainable_params'] / 1e6:.2f}M trainable of {report['total_params'] / 1e6:.1f}M "
          f"({report['trainable_pct']:.2f}%), features {'causal' if causal else 'bidirectional'}, dim {clf.hidden_dim}")
    pad_id = tok.token_to_id("<|pad|>")
    train_ds = nd.NERDataset(tr_t, tr_g, tok, tag2id, cfg.window, cfg.stride)
    val_ds = nd.NERDataset(va_t, va_g, tok, tag2id, cfg.window, cfg.stride) if va_t else None
    print(f"train {len(tr_t)} docs / {len(train_ds.examples)} windows, val {len(va_t)} docs | alignment {train_ds.align_stats}")

    weight = None
    if cfg.class_weighting == "inv_sqrt":
        c = train_ds.label_counts(len(tags)).astype(np.float64) + 1.0
        weight = torch.tensor((c.sum() / c) ** 0.5, dtype=torch.float32, device=device)
        weight = weight / weight.mean()

    teacher = None
    if cfg.kd_teacher:
        teacher, *_ = load_ner_run(cfg.kd_teacher, device)
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"KD teacher: {cfg.kd_teacher} ({param_report(teacher)['total_params'] / 1e6:.1f}M params)")

    head_params = list(clf.head.parameters())
    head_ids = {id(p) for p in head_params}
    body = [p for p in clf.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([dict(params=body, lr=cfg.lr), dict(params=head_params, lr=cfg.head_lr)],
                            weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds.examples) / cfg.batch_size)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)

    log = open(os.path.join(run_dir, "log.csv"), "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "train_loss", "val_macro_f1", "val_macro_p", "val_macro_r", "val_micro_f1", "seconds"])
    best, best_scores, rng = -1.0, None, random.Random(cfg.seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_time = 0.0
    for ep in range(1, cfg.epochs + 1):
        clf.train()
        te = time.time()
        tot = n = 0
        for _, ids, lab in _batches(train_ds, train_ds.examples, cfg.batch_size, pad_id, True, rng):
            ids, lab = ids.to(device), lab.to(device)
            with amp:
                logits = clf(ids).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), lab.view(-1), weight=weight, ignore_index=-100)
            if teacher is not None:
                with torch.no_grad(), amp:
                    t_logits = teacher(ids).float()
                m = lab.view(-1) != -100
                T = cfg.kd_temperature
                kl = F.kl_div(F.log_softmax(logits.view(-1, logits.size(-1))[m] / T, -1),
                              F.softmax(t_logits.view(-1, t_logits.size(-1))[m] / T, -1), reduction="batchmean")
                loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * T * T * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in clf.parameters() if p.requires_grad], cfg.grad_clip)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        train_time += time.time() - te
        row = [ep, round(tot / max(n, 1), 5)]
        if val_ds is not None:
            preds = predict_docs(clf, val_ds, va_t, tags, causal, cfg.batch_size * 2, device, pad_id,
                                 cfg.decode, cfg.window_agg)
            s = strict_scores(preds, va_g, labels)
            row += [round(s["macro_f1"], 5), round(s["macro_p"], 5), round(s["macro_r"], 5), round(s["micro_f1"], 5)]
            print(f"epoch {ep}: loss {row[1]:.4f} | val strict macro-F1 {s['macro_f1']:.4f} (P {s['macro_p']:.3f}, "
                  f"R {s['macro_r']:.3f}) | {time.time() - te:.0f}s")
            if s["macro_f1"] > best:
                best, best_scores = s["macro_f1"], s
                torch.save(dict(state=clf.state_dict(), epoch=ep, scores=s), os.path.join(run_dir, "best.pt"))
                json.dump(preds, open(os.path.join(run_dir, "val_pred.json"), "w"))
        else:
            row += ["", "", "", ""]
            print(f"epoch {ep}: loss {row[1]:.4f} | {time.time() - te:.0f}s")
        lw.writerow(row + [round(time.time() - te, 1)]); log.flush()
    log.close()
    if val_ds is None:
        torch.save(dict(state=clf.state_dict(), epoch=cfg.epochs, scores=None), os.path.join(run_dir, "best.pt"))

    results = dict(run=cfg.run_name, method=cfg.method, phase1_run=os.path.basename(cfg.phase1_run.rstrip("/")),
                   kd_teacher=os.path.basename(cfg.kd_teacher.rstrip("/")) if cfg.kd_teacher else "",
                   best_val_macro_f1=best if val_ds is not None else None,
                   best_val_scores=best_scores, train_seconds=round(train_time, 1),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None,
                   total_seconds=round(time.time() - t0, 1), **report)
    json.dump(results, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    if best_scores:
        print("\nbest epoch on val:\n" + format_scores(best_scores))
    return results


def eval_decoding(run_dir, variants=(("argmax", "owner"), ("viterbi", "owner"), ("argmax", "mean"), ("viterbi", "mean")),
                  device="cuda", bs=16):
    """Score span-decoding variants on the run's own held-out val docs, re-using one forward pass (no retraining).
    Appends rows to run_dir/decoding_eval.jsonl and returns them."""
    clf, tok, mtype, cfg, (labels, tags, tag2id) = load_ner_run(run_dir, device)
    if not cfg.n_val_docs:
        raise ValueError("this run trained on all docs (n_val_docs=0): no held-out docs to compare decoders on")
    texts, golds = nd.load_folds(cfg.data_dir, [1, 2])
    _, (va_t, va_g) = nd.split_docs(texts, golds, cfg.n_val_docs, cfg.seed)
    ds = nd.NERDataset(va_t, va_g, tok, tag2id, cfg.window, cfg.stride)
    win_lp = window_logprobs(clf, ds, bs, device, tok.token_to_id("<|pad|>"))
    rows = []
    for dec, agg in variants:
        s = strict_scores(decode_docs(win_lp, ds, va_t, tags, mtype == "decoder", dec, agg), va_g, labels)
        row = dict(run=cfg.run_name, decode=dec, window_agg=agg, macro_f1=s["macro_f1"], macro_p=s["macro_p"],
                   macro_r=s["macro_r"], micro_f1=s["micro_f1"])
        rows.append(row)
        print(f"{dec:8s} {agg:6s} strict macro-F1 {s['macro_f1']:.4f}  P {s['macro_p']:.4f}  R {s['macro_r']:.4f}")
    with open(os.path.join(run_dir, "decoding_eval.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def predict_test(run_dir, out_dir=None, device="cuda", bs=16, decode=None, window_agg=None):
    """Predict fold3 with a trained run and write prediction.json + submission.zip (file at the zip root).
    decode / window_agg default to the run's config; pass them to override (e.g. the variant you chose)."""
    from ..submit_json import write_json_submission
    clf, tok, mtype, cfg, (labels, tags, tag2id) = load_ner_run(run_dir, device)
    decode, window_agg = decode or cfg.decode, window_agg or cfg.window_agg
    texts, _ = nd.load_folds(cfg.data_dir, [3])
    ds = nd.NERDataset(texts, {}, tok, tag2id, cfg.window, cfg.stride)
    t0 = time.time()
    preds = predict_docs(clf, ds, texts, tags, mtype == "decoder", bs, device, tok.token_to_id("<|pad|>"),
                         decode, window_agg)
    secs = time.time() - t0
    for d, ents in preds.items():
        for s, e, l in ents:
            assert 0 <= s < e <= len(texts[d]) and l in labels
    out = out_dir or os.path.join(run_dir, f"submission_{decode}_{window_agg}")
    zip_path = write_json_submission(preds, expected_ids=set(texts), out_dir=out)
    n = sum(len(v) for v in preds.values())
    print(f"{n} entities over {len(preds)} test docs ({secs:.1f}s inference, {secs / len(preds):.2f}s/doc)")
    return zip_path, preds
