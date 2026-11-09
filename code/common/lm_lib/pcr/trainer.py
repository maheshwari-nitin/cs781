"""Shared Phase-1 representations, supervised retrieval adaptation, score KD and exact top-7 search.

No new backbone/head: reuse ner.hidden_states, existing adapters, and mean/last feature pooling.
Candidate embeddings are computed separately for each official split. Test labels are never accessed.
"""
import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from ..amp import autocast, describe, grad_scaler
from ..ner.model import apply_method, enable_grad_checkpointing, hidden_states
from ..submit_json import write_json_submission
from ..trainer import load_trained
from . import data
from .metrics import retrieval_scores, validate_predictions


@dataclass
class PCRConfig:
    phase1_run: str = ""
    phase1_ckpt: str = "best"
    method: str = "full"  # full | partial | lorma | lora
    method_kwargs: dict = field(default_factory=dict)
    data_dir: str = "/content/p7"
    out_dir: str = "/content/drive/MyDrive/cs781/pcr_runs"
    run_name: str = ""
    chunk_len: int = 256
    max_chunks: int = 2
    strategy: str = "first_last"  # first | last | first_last | spread
    token_pooling: str = "mean"  # mean | last
    chunk_agg: str = "mean"  # mean | max
    normalize_text: bool = True
    negatives: int = 4
    max_positives: int = 2
    negative_sampling: str = "random"  # random | tfidf (mixed hard/random)
    hard_negative_fraction: float = 0.5
    hard_negative_pool_size: int = 64
    resample_negatives: bool = False
    temperature: float = 0.05
    n_train: int = 0
    n_dev: int = 0  # 0 = every dev query; always rank against the FULL dev pool
    embed_batch_size: int = 16
    search_batch_size: int = 64
    train_batch_size: int = 1  # queries per physical microbatch, with independent candidate sets
    train_chunk_batch_size: int = 16  # equal-length case chunks per backbone forward
    grad_accum: int = 4
    auto_batch: bool = False
    batch_mem_frac: float = 0.90
    effective_batch_queries: int = 0  # 0 preserves train_batch_size * grad_accum
    max_train_batch: int = 32
    max_embed_batch: int = 1024
    batch_calibration: dict = field(default_factory=dict)
    grad_checkpoint: bool = True
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 0
    early_stopping_patience: int = 0  # number of dev evaluations without improvement; 0 disables
    kd_teacher: str = ""
    kd_alpha: float = 0.5
    kd_temperature: float = 2.0
    stop_at: str = ""
    seed: int = 1337

    def __post_init__(self):
        if not self.run_name or os.path.basename(self.run_name) != self.run_name or self.run_name in (".", ".."):
            raise ValueError("Provide a new run_name containing one directory name")
        if self.method not in ("full", "partial", "lorma", "lora"):
            raise ValueError("Unsupported retrieval adaptation method")
        if min(self.chunk_len, self.max_chunks, self.negatives, self.embed_batch_size,
               self.search_batch_size, self.grad_accum, self.epochs, self.train_batch_size,
               self.train_chunk_batch_size, self.max_train_batch, self.max_embed_batch) < 1:
            raise ValueError("Chunk/batch/epoch/negative settings must be positive")
        if min(self.n_train, self.n_dev, self.max_positives, self.eval_every, self.effective_batch_queries) < 0:
            raise ValueError("Subset, positive cap and evaluation settings cannot be negative")
        if self.negative_sampling not in ("random", "tfidf") or not 0 <= self.hard_negative_fraction <= 1:
            raise ValueError("Invalid negative sampling configuration")
        if self.hard_negative_pool_size < 1 or self.early_stopping_patience < 0:
            raise ValueError("Invalid hard-negative pool size or early-stopping patience")
        if self.kd_teacher and self.resample_negatives:
            raise ValueError("Cached KD scores require fixed pairs: disable resample_negatives for KD")
        if self.strategy not in ("first", "last", "first_last", "spread"):
            raise ValueError("Unknown chunk selection strategy")
        if self.token_pooling not in ("mean", "last") or self.chunk_agg not in ("mean", "max"):
            raise ValueError("Unknown pooling method")
        if self.temperature <= 0 or self.kd_temperature <= 0 or not 0 <= self.kd_alpha <= 1:
            raise ValueError("Invalid similarity/KD temperature or weight")
        if not 0 <= self.warmup_frac < 1:
            raise ValueError("Invalid warmup fraction")
        if self.method == "partial" and self.method_kwargs.get("k", 4) < 1:
            raise ValueError("partial k must be positive")
        if not 0 < self.batch_mem_frac < 1:
            raise ValueError("batch_mem_frac must be in (0, 1)")


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chunks(doc, tok, cfg):
    text = " ".join(doc["text"].split()) if cfg.normalize_text else doc["text"]
    ids = tok.encode(text).ids
    budget = cfg.chunk_len * cfg.max_chunks
    if len(ids) > budget:
        if cfg.strategy == "spread":
            starts = [round(i * (len(ids) - cfg.chunk_len) / max(cfg.max_chunks - 1, 1))
                      for i in range(cfg.max_chunks)]
            return [ids[s:s + cfg.chunk_len] for s in starts]
        if cfg.strategy == "first":
            ids = ids[:budget]
        elif cfg.strategy == "last":
            ids = ids[-budget:]
        else:
            head = (budget + 1) // 2
            ids = ids[:head] + (ids[-(budget - head):] if budget > head else [])
    # Empty text still has a representation and retains its ID in the candidate pool/test set.
    ids = ids or [tok.token_to_id("<|bos|>")]
    return [ids[i:i + cfg.chunk_len] for i in range(0, len(ids), cfg.chunk_len)]


def _pool_tokens(h, cfg):
    return h.float().mean(1) if cfg.token_pooling == "mean" else h[:, -1].float()


def _pool_chunks(vectors, cfg):
    matrix = torch.stack(vectors)
    vector = matrix.mean(0) if cfg.chunk_agg == "mean" else matrix.max(0).values
    return F.normalize(vector, dim=-1)


def case_vector(model, token_chunks, cfg, device):
    vectors = []
    for tokens in token_chunks:
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with autocast(device):
            vectors.append(_pool_tokens(hidden_states(model, ids), cfg)[0])
    return _pool_chunks(vectors, cfg)


def embed_token_batch(model, token_rows, cfg, device):
    """Equal-length chunks only; called with gradients for training and no_grad for inference."""
    ids = torch.tensor(token_rows, dtype=torch.long, device=device)
    with autocast(device):
        return _pool_tokens(hidden_states(model, ids), cfg)


def case_vectors(model, cases, cfg, device):
    """Differentiable batched version of case_vector, without padding or cross-query negatives."""
    buckets = defaultdict(list)
    vectors = [[None] * len(parts) for parts in cases]
    for i, parts in enumerate(cases):
        for j, tokens in enumerate(parts):
            buckets[len(tokens)].append((i, j, tokens))
    for length in sorted(buckets, reverse=True):
        entries = buckets[length]
        for start in range(0, len(entries), cfg.train_chunk_batch_size):
            batch = entries[start:start + cfg.train_chunk_batch_size]
            pooled = embed_token_batch(model, [row[2] for row in batch], cfg, device)
            for (i, j, _), vector in zip(batch, pooled):
                vectors[i][j] = vector
    return torch.stack([_pool_chunks(parts, cfg) for parts in vectors])


def ranking_batch_losses(model, pairs, query_tokens, candidate_tokens, cfg, device,
                         kd=None, teacher_temperature=None):
    cases = []
    for pair in pairs:
        cases.append(query_tokens[pair["query_id"]])
        cases.extend(candidate_tokens[i] for i in pair["candidate_ids"])
    vectors = case_vectors(model, cases, cfg, device)
    losses, cursor = [], 0
    for pair in pairs:
        n = len(pair["candidate_ids"])
        q, c = vectors[cursor], vectors[cursor + 1:cursor + 1 + n]
        cursor += n + 1
        logits = (c @ q) / cfg.temperature
        loss = -F.log_softmax(logits, dim=0)[:pair["n_positive"]].mean()
        if kd is not None:
            target = kd[pair["query_id"]]
            if target["candidate_ids"] != pair["candidate_ids"]:
                raise ValueError("Teacher/student candidate order differs")
            t = cfg.kd_temperature
            teacher_logits = torch.tensor(target["scores"], device=device) / teacher_temperature
            distill = F.kl_div(F.log_softmax(logits / t, dim=0),
                              F.softmax(teacher_logits / t, dim=0), reduction="sum") * t * t
            loss = (1 - cfg.kd_alpha) * loss + cfg.kd_alpha * distill
        losses.append(loss)
    return torch.stack(losses)


def build_retriever(cfg, device):
    model, tok, p1cfg, mtype = load_trained(cfg.phase1_run, cfg.phase1_ckpt, device="cpu")
    base_params = sum(p.numel() for p in model.parameters())
    limit = p1cfg.src_len if mtype == "encdec" else p1cfg.seq_len
    if cfg.chunk_len > limit:
        raise ValueError(f"chunk_len {cfg.chunk_len} exceeds pretrained feature context {limit}")
    if tok.token_to_id("<|bos|>") is None:
        raise ValueError("Tokenizer must define <|bos|>")
    # apply_method already implements the repository's feature-stack freezing and adapter contract.
    wrapper = torch.nn.Module()
    wrapper.backbone, wrapper.head = model.float(), torch.nn.Identity()
    apply_method(wrapper, cfg.method, **cfg.method_kwargs)
    model.to(device).eval()
    # B uses encoder features: freeze unused decoder/LM-only parameters rather than training a decoder.
    probe = [[tok.token_to_id("<|bos|>")] * min(8, cfg.chunk_len)]
    case_vector(model, probe, cfg, device).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None and ".experts." not in name:
            p.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if cfg.grad_checkpoint:
        enable_grad_checkpointing(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise ValueError("No trainable feature parameters; check adapter targets")
    return model, tok, dict(base_params=base_params, total_params=sum(p.numel() for p in model.parameters()),
                            trainable_params=trainable, trainable_pct=100 * trainable / base_params)


@torch.no_grad()
def encode_corpus(model, tok, docs, cfg, device):
    """Batch ONLY equal-length chunks: no padding, no truncation to other documents' lengths."""
    model.eval()
    buckets, vectors = defaultdict(list), [[] for _ in docs]
    for i, doc in enumerate(docs):
        for tokens in chunks(doc, tok, cfg):
            buckets[len(tokens)].append((i, tokens))
    total = sum(len(v) for v in buckets.values())
    done = 0
    for length in sorted(buckets, reverse=True):
        entries = buckets[length]
        start = 0
        while start < len(entries):
            batch = entries[start:start + cfg.embed_batch_size]
            try:
                pooled = embed_token_batch(model, [tokens for _, tokens in batch], cfg, device).cpu()
            except torch.cuda.OutOfMemoryError as error:
                if device != "cuda" or len(batch) == 1:
                    raise
                import traceback
                traceback.clear_frames(error.__traceback__)
                gc.collect(); torch.cuda.empty_cache()
                cfg.embed_batch_size = max(1, len(batch) // 2)
                print(f"Embedding OOM: retrying the same chunks at batch {cfg.embed_batch_size}", flush=True)
                continue
            for (i, _), vector in zip(batch, pooled):
                vectors[i].append(vector)
            done += len(batch)
            start += len(batch)
            if done == total or done % 1000 < len(batch):
                print(f"encoded chunks {done}/{total}", flush=True)
    result = torch.stack([_pool_chunks(v, cfg) for v in vectors])
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite case embeddings")
    return result


def rank_embeddings(query_ids, query_vectors, candidate_ids, candidate_vectors, batch_size=64):
    if len(candidate_ids) < 7 or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Retrieval needs at least seven unique candidates")
    # Canonical lexical ID order makes ties deterministic and preserves every ID character.
    order = sorted(range(len(candidate_ids)), key=lambda i: candidate_ids[i])
    ids = [candidate_ids[i] for i in order]
    c = candidate_vectors[order].float().cpu()
    out = {}
    for start in range(0, len(query_ids), batch_size):
        scores = query_vectors[start:start + batch_size].float().cpu() @ c.T
        if not torch.isfinite(scores).all():
            raise ValueError("Nonfinite retrieval scores")
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :7].tolist()
        for ident, row in zip(query_ids[start:start + batch_size], ranked):
            out[ident] = [ids[i] for i in row]
    validate_predictions(out, query_ids, candidate_ids)
    return out


def retrieve(model, tok, queries, pool, cfg, device, candidate_vectors=None):
    c = encode_corpus(model, tok, pool, cfg, device) if candidate_vectors is None else candidate_vectors
    q = encode_corpus(model, tok, queries, cfg, device)
    return rank_embeddings([d["id"] for d in queries], q, [d["id"] for d in pool], c, cfg.search_batch_size)


def _save(model, cfg, run_dir, metrics, step, epoch):
    keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
    state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in keep}
    path = os.path.join(run_dir, "best.pt")
    # Includes only changed feature parameters/buffers; frozen Phase-1 weights are reconstructed on load.
    torch.save(dict(state=state, metrics=metrics, step=step, epoch=epoch), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_pcr_run(run_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = PCRConfig(**read(os.path.join(run_dir, "config.json")))
    manifest = read(os.path.join(run_dir, "manifest.json"))
    if file_hash(os.path.join(cfg.phase1_run, cfg.phase1_ckpt + ".pt")) != manifest["phase1_sha256"]:
        raise ValueError("Original Phase-1 checkpoint changed; restore the original base weights")
    model, tok, report = build_retriever(cfg, device)
    if json.loads(tok.to_str()) != read(os.path.join(run_dir, "tokenizer.json")):
        raise ValueError("Phase-1 tokenizer changed since training")
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    live = dict(model.named_parameters()) | dict(model.named_buffers())
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if trainable - set(ck["state"]) or set(ck["state"]) - set(live):
        raise ValueError("Incomplete retrieval checkpoint")
    with torch.no_grad():
        for name, tensor in ck["state"].items():
            if live[name].shape != tensor.shape:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            live[name].copy_(tensor)
    return model.eval(), tok, cfg, report


def _teacher_targets(cfg, pairs, queries, pool, student_report, manifest, run_dir, device):
    teacher_manifest = read(os.path.join(cfg.kd_teacher, "manifest.json"))
    if teacher_manifest["train_fingerprint"] != manifest["train_fingerprint"]:
        raise ValueError("KD teacher must use the same official TRAIN queries and candidate pool")
    teacher, tok, tcfg, treport = load_pcr_run(cfg.kd_teacher, device)
    if student_report["base_params"] >= treport["base_params"]:
        raise ValueError("KD requires a smaller student Phase-1 backbone")
    try:
        needed = {v for pair in pairs for v in pair["candidate_ids"]}
        candidates = [d for d in pool if d["id"] in needed]
        qvec = encode_corpus(teacher, tok, queries, tcfg, device)
        cvec = encode_corpus(teacher, tok, candidates, tcfg, device)
        qmap = {d["id"]: v for d, v in zip(queries, qvec)}
        cmap = {d["id"]: v for d, v in zip(candidates, cvec)}
        targets = {p["query_id"]: dict(candidate_ids=p["candidate_ids"],
                   scores=[float(qmap[p["query_id"]] @ cmap[i]) for i in p["candidate_ids"]]) for p in pairs}
        dump(dict(teacher=cfg.kd_teacher, teacher_checkpoint_sha256=file_hash(os.path.join(cfg.kd_teacher, "best.pt")),
                  teacher_temperature=tcfg.temperature, train_fingerprint=manifest["train_fingerprint"],
                  targets=targets), os.path.join(run_dir, "kd_targets.json"))
    finally:
        del teacher
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return targets, tcfg.temperature, treport


def _train_pcr(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(f"Run exists: {run_dir}. Use a new name; optimizer resume is not implemented.")
    deadline = None
    if cfg.stop_at:
        from datetime import datetime
        timestamp = datetime.fromisoformat(cfg.stop_at)
        if timestamp.tzinfo is None:
            raise ValueError("stop_at must include timezone")
        deadline = timestamp.timestamp()
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    started = time.time()
    train_all, train_pool = data.load_split(cfg.data_dir, "train")
    dev_all, dev_pool = data.load_split(cfg.data_dir, "dev")
    if {q["id"] for q in train_all} & {q["id"] for q in dev_all}:
        raise ValueError("Training and dev query IDs overlap")
    if len(dev_pool) < 7:
        raise ValueError("Dev pool must contain at least seven candidates")
    train_queries = sorted(train_all, key=lambda d: d["id"])
    dev_queries = sorted(dev_all, key=lambda d: d["id"])
    if cfg.n_train and cfg.n_train < len(train_queries):
        train_queries = random.Random(cfg.seed).sample(train_queries, cfg.n_train)
    if cfg.n_dev and cfg.n_dev < len(dev_queries):
        dev_queries = random.Random(cfg.seed).sample(dev_queries, cfg.n_dev)
    hard_candidates = None
    if cfg.negative_sampling == "tfidf":
        from .mining import mine_tfidf_negatives
        hard_candidates = mine_tfidf_negatives(train_queries, train_pool, cfg.hard_negative_pool_size)
    def make_pairs(epoch=1):
        seed = cfg.seed + (epoch - 1) * 10007 if cfg.resample_negatives else cfg.seed
        return data.sample_pairs(train_queries, train_pool, cfg.negatives, cfg.max_positives, seed,
                                 hard_candidates, cfg.hard_negative_fraction)
    pairs, skipped = make_pairs()
    if cfg.auto_batch:
        from .batching import calibrate_batches
        cfg = calibrate_batches(cfg, device=device,
                                candidate_count=max(len(p["candidate_ids"]) for p in pairs))
    model, tok, report = build_retriever(cfg, device)
    manifest = dict(train_fingerprint=data.fingerprint(train_all, train_pool),
                    dev_fingerprint=data.fingerprint(dev_all, dev_pool),
                    train_query_ids=[q["id"] for q in train_all], dev_query_ids=[q["id"] for q in dev_all],
                    used_train_query_ids=[p["query_id"] for p in pairs], eval_query_ids=[q["id"] for q in dev_queries],
                    skipped_train_query_ids=skipped, train_candidates=len(train_pool), dev_candidates=len(dev_pool),
                    phase1_sha256=file_hash(os.path.join(cfg.phase1_run, cfg.phase1_ckpt + ".pt")))
    os.makedirs(run_dir, exist_ok=True)
    dump(asdict(cfg), os.path.join(run_dir, "config.json")); dump(manifest, os.path.join(run_dir, "manifest.json"))
    dump(pairs, os.path.join(run_dir, "train_pairs.json")); tok.save(os.path.join(run_dir, "tokenizer.json"))
    if hard_candidates is not None:
        dump(hard_candidates, os.path.join(run_dir, "train_hard_negative_pool.json"))
    if cfg.batch_calibration:
        dump(cfg.batch_calibration, os.path.join(run_dir, "batch_calibration.json"))
    print(f"{cfg.run_name}: {len(pairs)} train queries, {len(dev_queries)} dev queries / {len(dev_pool)} dev candidates | {describe(device)}", flush=True)
    kd, teacher_temperature, teacher_report = None, None, None
    if cfg.kd_teacher:
        used = set(manifest["used_train_query_ids"])
        kd, teacher_temperature, teacher_report = _teacher_targets(
            cfg, pairs, [q for q in train_queries if q["id"] in used], train_pool, report, manifest, run_dir, device)
    query_tokens = {q["id"]: chunks(q, tok, cfg) for q in train_queries}
    needed = {v for p in pairs for v in p["candidate_ids"]}
    candidate_tokens = {d["id"]: chunks(d, tok, cfg) for d in train_pool if d["id"] in needed}
    pool_by_id = {d["id"]: d for d in train_pool}
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    group_size = cfg.train_batch_size * cfg.grad_accum
    total = math.ceil(len(pairs) / group_size) * cfg.epochs
    warm = int(cfg.warmup_frac * total)
    def lr_scale(step):
        return (step + 1) / warm if step < warm else max(0.0, (total - step) / max(1, total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    scaler = grad_scaler(device)
    steps, best, stopped, last_eval = 0, -1.0, False, -1
    stale_evals, stop_reason, best_epoch, best_step = 0, None, 0, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "step", "train_loss", "dev_micro_f1_at7", "dev_micro_precision_at7", "dev_micro_recall_at7", "seconds"])
        def evaluate(epoch, loss):
            nonlocal best, last_eval, stale_evals, stop_reason, best_epoch, best_step
            predictions = retrieve(model, tok, dev_queries, dev_pool, cfg, device)
            metrics = retrieval_scores(predictions, dev_queries)
            writer.writerow([epoch, steps, loss, metrics["micro_f1_at7"], metrics["micro_precision_at7"],
                             metrics["micro_recall_at7"], time.time() - started]); log.flush()
            print(f"[dev] step {steps}: micro-F1@7 {metrics['micro_f1_at7']:.4f}", flush=True)
            if metrics["micro_f1_at7"] > best:
                best = metrics["micro_f1_at7"]
                stale_evals, best_epoch, best_step = 0, epoch, steps
                _save(model, cfg, run_dir, metrics, steps, epoch)
                dump(metrics, os.path.join(run_dir, "dev_metrics.json"))
                dump(predictions, os.path.join(run_dir, "dev_predictions.json"))
            else:
                stale_evals += 1
            if cfg.early_stopping_patience and stale_evals >= cfg.early_stopping_patience and stop_reason is None:
                stop_reason = "dev_patience"
                print(f"Early stopping after {stale_evals} dev evaluations without improvement", flush=True)
            last_eval = steps
            model.train()
        for epoch in range(1, cfg.epochs + 1):
            if cfg.resample_negatives:
                pairs, _ = make_pairs(epoch)
                dump(pairs, os.path.join(run_dir, f"train_pairs_epoch_{epoch:03d}.json"))
                for ident in {v for p in pairs for v in p["candidate_ids"]} - set(candidate_tokens):
                    candidate_tokens[ident] = chunks(pool_by_id[ident], tok, cfg)
            order = list(range(len(pairs))); random.Random(cfg.seed + epoch).shuffle(order)
            total_loss, count = 0.0, 0
            model.train()
            for start in range(0, len(order), group_size):
                if deadline and time.time() >= deadline:
                    stopped = True
                    stop_reason = "deadline"
                    break
                group = order[start:start + group_size]
                opt.zero_grad(set_to_none=True)
                for micro_start in range(0, len(group), cfg.train_batch_size):
                    micro = [pairs[i] for i in group[micro_start:micro_start + cfg.train_batch_size]]
                    losses = ranking_batch_losses(model, micro, query_tokens, candidate_tokens,
                                                   cfg, device, kd, teacher_temperature)
                    scaler.scale(losses.sum() / len(group)).backward()
                    total_loss += losses.detach().sum().item(); count += len(micro)
                    del losses
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                old = scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale() >= old:
                    steps += 1; sched.step()
                    if hasattr(model, "post_step"):
                        model.post_step()
                opt.zero_grad(set_to_none=True)
                if steps == 1 or steps % 20 == 0:
                    print(f"epoch {epoch}/{cfg.epochs} step {steps}/{total} loss {total_loss / count:.4f}", flush=True)
                if cfg.eval_every and steps and steps % cfg.eval_every == 0 and steps != last_eval:
                    evaluate(epoch, total_loss / count)
                    if stop_reason == "dev_patience":
                        stopped = True
                        break
            if steps != last_eval:
                evaluate(epoch, total_loss / max(count, 1))
            if stop_reason == "dev_patience":
                stopped = True
            if stopped:
                break
    result = dict(run=cfg.run_name, phase1_run=cfg.phase1_run, method=cfg.method, kd_teacher=cfg.kd_teacher,
                  teacher_params=teacher_report, n_train=len(pairs), n_dev=len(dev_queries),
                  best_dev_micro_f1_at7=best, best_dev_metrics=read(os.path.join(run_dir, "dev_metrics.json")),
                  optimizer_steps=steps, stopped_early=stopped, total_seconds=time.time() - started,
                  stop_reason=stop_reason, best_epoch=best_epoch, best_step=best_step, completed_epochs=epoch,
                  negative_sampling=cfg.negative_sampling, resample_negatives=cfg.resample_negatives,
                  train_batch_size=cfg.train_batch_size, train_chunk_batch_size=cfg.train_chunk_batch_size,
                  embed_batch_size=cfg.embed_batch_size, effective_batch_queries=group_size,
                  peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None, **report)
    dump(result, os.path.join(run_dir, "results.json"))
    return result


def train_pcr(cfg):
    try:
        return _train_pcr(cfg)
    except BaseException as error:
        from ..rr.trainer import free_gpu_after_failure
        free_gpu_after_failure(error)
        raise


def predict_test(run_dir, data_dir=None, out_dir=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg, _ = load_pcr_run(run_dir, device)
    queries, pool = data.load_split(data_dir or cfg.data_dir, "test", labeled=False)
    start = time.time()
    cache_key = dict(pool_fingerprint=data.fingerprint([], pool),
                     checkpoint_sha256=file_hash(os.path.join(run_dir, "best.pt")),
                     representation={k: getattr(cfg, k) for k in
                         ("chunk_len", "max_chunks", "strategy", "token_pooling", "chunk_agg", "normalize_text")})
    cache_path = os.path.join(run_dir, "test_candidate_embeddings.pt")
    vectors = None
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached["key"] == cache_key and cached["candidate_ids"] == [d["id"] for d in pool]:
            vectors = cached["vectors"]
    if vectors is None:
        vectors = encode_corpus(model, tok, pool, cfg, device)
        torch.save(dict(key=cache_key, candidate_ids=[d["id"] for d in pool], vectors=vectors), cache_path + ".tmp")
        os.replace(cache_path + ".tmp", cache_path)
    predictions = retrieve(model, tok, queries, pool, cfg, device, vectors)
    validate_predictions(predictions, [q["id"] for q in queries], [d["id"] for d in pool])
    out_dir = out_dir or os.path.join(run_dir, "submission")
    path = write_json_submission(predictions, {q["id"] for q in queries}, out_dir)
    dump(dict(n_queries=len(queries), n_candidates=len(pool), inference_seconds=time.time() - start,
              candidate_cache_key=cache_key), os.path.join(out_dir, "inference.json"))
    return path, predictions
