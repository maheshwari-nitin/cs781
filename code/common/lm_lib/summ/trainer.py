"""Supervised summary-only loss and sequence KD on existing Phase-1 generators.

One document per forward avoids padding leakage in models without attention-mask interfaces.
Gradient accumulation supplies effective batches. All architecture/adapter implementations are reused.
"""
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from ..amp import autocast, describe, grad_scaler
from ..generate import apply_repetition_penalty, sample_next
from ..ner.model import enable_grad_checkpointing, feature_blocks
from ..submit_json import write_json_submission
from ..trainer import load_trained
from . import data
from .metrics import score_summaries


@dataclass
class SummConfig:
    phase1_run: str = ""
    phase1_ckpt: str = "best"
    method: str = "lora"  # lora | lorma | qlora | full | partial
    method_kwargs: dict = field(default_factory=dict)
    data_dir: str = "/content/p6"
    out_dir: str = "/content/drive/MyDrive/cs781/summ_runs"
    run_name: str = ""
    val_fraction: float = 0.05
    split_seed: int = 1337
    n_train: int = 0
    n_val_generate: int = 32  # 0 = all held-out docs; saved exact IDs are comparable across runs
    source_tokens: int = 640  # includes BOS and document/summary markers
    target_tokens: int = 256  # includes EOT; explicit training target truncation
    strategy: str = "first_last"
    normalize_text: bool = True
    grad_accum: int = 8
    grad_checkpoint: bool = False
    epochs: int = 3
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 0  # optimizer steps, 0 = epoch-end only
    selection_metric: str = "rougeL_f1"  # or bertscore_f1; never select by test references
    bert_model: str = "roberta-large"
    bert_device: str = "cpu"
    max_new_tokens: int = 256
    min_new_tokens: int = 16
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    kd_teacher: str = ""  # trained summarization run; generate TRAIN summaries only
    kd_alpha: float = 0.5  # mixture of reference CE and teacher-summary CE
    stop_at: str = ""  # timezone-aware ISO timestamp, finish an optimizer group then save/evaluate
    seed: int = 1337

    def __post_init__(self):
        if not self.run_name or os.path.basename(self.run_name) != self.run_name or self.run_name in (".", ".."):
            raise ValueError("Provide a new run_name containing one directory name")
        if self.method not in ("lora", "lorma", "qlora", "full", "partial"):
            raise ValueError("Unsupported adaptation method")
        if not 0 < self.val_fraction < 1 or min(self.n_train, self.n_val_generate, self.eval_every) < 0:
            raise ValueError("Invalid split, subset or evaluation configuration")
        if min(self.source_tokens, self.target_tokens, self.grad_accum, self.epochs, self.max_new_tokens) < 1:
            raise ValueError("Token budgets, epochs and accumulation must be positive")
        if self.target_tokens < 2 or self.max_new_tokens > self.target_tokens:
            raise ValueError("Require 2 <= target_tokens and max_new_tokens <= target_tokens")
        if not 0 <= self.min_new_tokens < self.max_new_tokens:
            raise ValueError("min_new_tokens must be below max_new_tokens")
        if self.strategy not in ("first", "last", "first_last"):
            raise ValueError("Unknown context selection")
        if self.selection_metric not in ("rougeL_f1", "bertscore_f1"):
            raise ValueError("Select by rougeL_f1 or bertscore_f1")
        if not 0 <= self.kd_alpha <= 1 or not 0 <= self.warmup_frac < 1:
            raise ValueError("Invalid KD weight or warmup")
        if self.repetition_penalty <= 0 or not 0 < self.top_p <= 1 or min(self.top_k, self.no_repeat_ngram_size) < 0:
            raise ValueError("Invalid decoding controls")


def dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _quantize_lora_bases(model):
    """NF4 only for existing LoRA base projections. Tied embeddings/output stay full precision.

    Reference: https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit
    This optional path is restricted to architecture A; custom FLA/MoE kernels are not assumed compatible.
    """
    from bitsandbytes.nn import Linear4bit, Params4bit
    from adapters.lora import LoRALinear
    compute = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    count = 0
    for module in list(model.modules()):
        if isinstance(module, LoRALinear):
            base = module.base_layer
            quant = Linear4bit(base.in_features, base.out_features, bias=base.bias is not None,
                               compute_dtype=compute, compress_statistics=True, quant_type="nf4")
            quant.weight = Params4bit(base.weight.detach().cpu(), requires_grad=False,
                                     compress_statistics=True, quant_type="nf4")
            if base.bias is not None:
                quant.bias = torch.nn.Parameter(base.bias.detach().cpu(), requires_grad=False)
            module.base_layer = quant
            count += 1
    if not count:
        raise ValueError("QLoRA matched no projections")
    return count


def build_model(cfg, device):
    # CPU first: QLoRA quantizes directly on transfer without first filling CUDA with fp32 weights.
    model, tok, p1cfg, model_type = load_trained(cfg.phase1_run, cfg.phase1_ckpt, device="cpu")
    model.float()
    base_params = sum(p.numel() for p in model.parameters())
    if any(tok.token_to_id(t) is None for t in ("<|bos|>", "<|pad|>", "<|endoftext|>")):
        raise ValueError("Phase-1 tokenizer must define BOS, PAD and endoftext")
    if model_type == "decoder" and cfg.source_tokens + cfg.target_tokens > p1cfg.seq_len:
        raise ValueError(f"source_tokens + target_tokens exceeds context {p1cfg.seq_len}")
    if model_type == "encdec" and (cfg.source_tokens > p1cfg.src_len or
                                  cfg.target_tokens > p1cfg.seq_len - p1cfg.src_len):
        raise ValueError("Source/target budgets exceed pretrained encoder/decoder lengths")
    # Validate marker overhead before any long data preparation.
    data.source_ids(dict(document="preflight"), tok, cfg)
    if cfg.method == "full":
        model.requires_grad_(True)
    elif cfg.method == "partial":
        k = cfg.method_kwargs.get("k", 4)
        if k < 1:
            raise ValueError("partial k must be positive")
        model.requires_grad_(False)
        stacks = [feature_blocks(model)]
        if model_type == "encdec":
            stacks.append(model.decoder if hasattr(model, "decoder") else model.tf.decoder.layers)
        for stack in stacks:
            for block in list(stack)[-k:]:
                block.requires_grad_(True)
        for name, module in model.named_children():
            if "norm" in name:
                module.requires_grad_(True)
    else:
        importlib.import_module(f"adapters.{'lora' if cfg.method == 'qlora' else cfg.method}").apply(model, **cfg.method_kwargs)
    if cfg.method == "qlora":
        if device != "cuda" or p1cfg.arch != "A_qwen_dense":
            raise ValueError("This QLoRA integration requires CUDA and architecture A_qwen_dense")
        _quantize_lora_bases(model)
    model.to(device)
    if cfg.grad_checkpoint:
        enable_grad_checkpointing(model)
        if model_type == "encdec":
            # B needs checkpointing on both generation stacks, not just its encoder.
            import torch.utils.checkpoint as cp
            for block in model.decoder:
                if getattr(block, "_summ_ckpt", False):
                    continue
                original = block.forward
                def wrapped(*args, _fwd=original, **kw):
                    return cp.checkpoint(_fwd, *args, use_reentrant=False, **kw) if torch.is_grad_enabled() else _fwd(*args, **kw)
                block.forward, block._summ_ckpt = wrapped, True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not n_trainable:
        raise ValueError("No trainable parameters; check adapter targets")
    report = dict(base_params=base_params, trainable_params=n_trainable,
                  trainable_pct=100 * n_trainable / base_params, quantized=cfg.method == "qlora")
    return model, tok, model_type, report


def summary_loss(model, example, model_type, device):
    x = torch.tensor([example["x"]], dtype=torch.long, device=device)
    y = torch.tensor([example["y"]], dtype=torch.long, device=device)
    with autocast(device):
        if model_type == "encdec":
            src = torch.tensor([example["src"]], dtype=torch.long, device=device)
            logits = model(src, x)[0]
        else:
            logits = model(x)[0]
        # Compute CE ourselves: prompt positions are masked; backbone-specific LM/MTP losses are not used.
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100, reduction="sum"), int((y != -100).sum())


class _EndSummary(Exception):
    pass


@torch.no_grad()
def generate_one(model, tok, model_type, doc, cfg, device):
    model.eval()
    seed = (cfg.seed + int(hashlib.sha256(doc["id"].encode()).hexdigest()[:8], 16)) % (2 ** 32)
    torch.manual_seed(seed)
    src = torch.tensor([data.source_ids(doc, tok, cfg)], dtype=torch.long, device=device)
    bos, eos, pad = [tok.token_to_id(t) for t in ("<|bos|>", "<|endoftext|>", "<|pad|>")]
    generated = []
    def pick(logits, context):
        logits = logits.float().clone()
        if generated:
            context = torch.tensor([generated], dtype=torch.long, device=device)
            logits = apply_repetition_penalty(logits, context, cfg.repetition_penalty)
        logits[:, [bos, pad]] = -float("inf")
        if len(generated) < cfg.min_new_tokens:
            logits[:, eos] = -float("inf")
        n = cfg.no_repeat_ngram_size
        if n and len(generated) >= n - 1:
            prefix = generated[-(n - 1):] if n > 1 else []
            banned = {generated[j + n - 1] for j in range(len(generated) - n + 1)
                      if generated[j:j + n - 1] == prefix}
            if banned:
                logits[:, list(banned)] = -float("inf")
        if not torch.isfinite(logits).any():
            raise ValueError("Decoding controls banned every token")
        nxt = sample_next(logits, cfg.temperature, cfg.top_k, cfg.top_p)
        if nxt.item() == eos:
            raise _EndSummary
        generated.append(nxt.item())
        return nxt
    with autocast(device):
        try:
            if model_type == "decoder" and hasattr(model, "generate_tokens"):
                # Reuse existing caches. An EOT callback exits before filling the remaining token budget.
                model.generate_tokens(src, cfg.max_new_tokens, pick)
            elif model_type == "decoder":
                seq = src
                for _ in range(cfg.max_new_tokens):
                    nxt = pick(model(seq)[0][:, -1], seq)
                    seq = torch.cat([seq, nxt[:, None]], 1)
            else:
                memory = model.encode(src)
                dec = torch.tensor([[bos]], dtype=torch.long, device=device)
                for _ in range(cfg.max_new_tokens):
                    nxt = pick(model.decode(memory, dec)[:, -1], dec)
                    dec = torch.cat([dec, nxt[:, None]], 1)
        except _EndSummary:
            pass
    text = " ".join(tok.decode(generated).split())
    return text


def generate_docs(model, tok, model_type, docs, cfg, device):
    predictions = {}
    start = time.time()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
    try:
        for i, doc in enumerate(docs, 1):
            predictions[doc["id"]] = generate_one(model, tok, model_type, doc, cfg, device)
            if i == 1 or i % 20 == 0 or i == len(docs):
                print(f"generated {i}/{len(docs)} ({time.time() - start:.0f}s)", flush=True)
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    return predictions


def _save(model, cfg, run_dir, meta):
    state = model.state_dict()
    if cfg.method != "full":
        # Preserve learned parameters AND mutable persistent buffers (e.g. router bias), never packed NF4 state.
        keep = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
        state = {n: v for n, v in state.items() if n in keep}
    state = {n: v.detach().cpu().clone() for n, v in state.items()}
    path = os.path.join(run_dir, "best.pt")
    torch.save(dict(state=state, partial=cfg.method != "full", **meta), path + ".tmp")
    os.replace(path + ".tmp", path)


def load_summ_run(run_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = SummConfig(**read(os.path.join(run_dir, "config.json")))
    manifest = read(os.path.join(run_dir, "split.json"))
    base_path = os.path.join(cfg.phase1_run, cfg.phase1_ckpt + ".pt")
    if file_hash(base_path) != manifest["phase1_sha256"]:
        raise ValueError("Phase-1 checkpoint changed since this run; restore its original weights")
    model, tok, mtype, report = build_model(cfg, device)
    if json.loads(tok.to_str()) != read(os.path.join(run_dir, "tokenizer.json")):
        raise ValueError("Phase-1 tokenizer changed since training")
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cpu", weights_only=False)
    if ck["partial"]:
        # Copy only learned tensors/buffers. Do not ask NF4 submodules to deserialize absent base weights.
        live = dict(model.named_parameters()) | dict(model.named_buffers())
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        if trainable - set(ck["state"]) or set(ck["state"]) - set(live):
            raise ValueError("Incomplete trained checkpoint")
        with torch.no_grad():
            for name, tensor in ck["state"].items():
                if live[name].shape != tensor.shape:
                    raise ValueError(f"Checkpoint tensor shape mismatch: {name}")
                live[name].copy_(tensor)
    else:
        model.load_state_dict(ck["state"], strict=True)
    return model.eval(), tok, cfg, mtype, report


def _kd_targets(cfg, model_report, train_docs, manifest, device, run_dir):
    teacher_manifest = read(os.path.join(cfg.kd_teacher, "split.json"))
    if (teacher_manifest["data_sha256"] != manifest["data_sha256"] or
            set(teacher_manifest["val_ids"]) != set(manifest["val_ids"])):
        raise ValueError("KD teacher must use the same original dataset and held-out split")
    teacher, tok, tcfg, mtype, report = load_summ_run(cfg.kd_teacher, device)
    if model_report["base_params"] >= report["base_params"]:
        raise ValueError("KD requires a smaller student Phase-1 backbone")
    try:
        # Sequence KD supports different tokenizer vocabularies: transfer TEXT, never shared token IDs.
        predictions = generate_docs(teacher, tok, mtype, train_docs, tcfg, device)
        if any(not s.strip() for s in predictions.values()):
            raise ValueError("Teacher generated empty training summaries; inspect its outputs before KD")
        dump(dict(teacher=cfg.kd_teacher,
                  teacher_checkpoint_sha256=file_hash(os.path.join(cfg.kd_teacher, "best.pt")),
                  data_sha256=manifest["data_sha256"], predictions=predictions), os.path.join(run_dir, "kd_targets.json"))
    finally:
        del teacher
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return predictions, report


def _train_summ(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    if os.path.isdir(run_dir) and os.listdir(run_dir):
        raise FileExistsError(f"Run already exists: {run_dir}. Use a fresh name; optimizer resume is not implemented.")
    deadline = None
    if cfg.stop_at:
        from datetime import datetime
        dt = datetime.fromisoformat(cfg.stop_at)
        if dt.tzinfo is None:
            raise ValueError("stop_at must include timezone")
        deadline = dt.timestamp()
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    start = time.time()
    quality = {}
    docs = data.load_split(os.path.join(cfg.data_dir, "train.json"), report=quality)
    train_docs, val_docs = data.split_train_val(docs, cfg.val_fraction, cfg.split_seed)
    all_train_ids = [d["id"] for d in train_docs]
    if cfg.n_train and cfg.n_train < len(train_docs):
        train_docs = random.Random(cfg.seed).sample(train_docs, cfg.n_train)
    eval_docs = sorted(val_docs, key=lambda d: d["id"])
    if cfg.n_val_generate:
        eval_docs = random.Random(cfg.split_seed).sample(eval_docs, min(len(eval_docs), cfg.n_val_generate))
    model, tok, mtype, report = build_model(cfg, device)
    manifest = dict(data_sha256=data.fingerprint(docs), data_quality=quality, train_pool_ids=all_train_ids,
                    train_ids=[d["id"] for d in train_docs], val_ids=[d["id"] for d in val_docs],
                    eval_ids=[d["id"] for d in eval_docs], split_seed=cfg.split_seed,
                    phase1_sha256=file_hash(os.path.join(cfg.phase1_run, cfg.phase1_ckpt + ".pt")))
    os.makedirs(run_dir, exist_ok=True)
    dump(asdict(cfg), os.path.join(run_dir, "config.json"))
    dump(manifest, os.path.join(run_dir, "split.json"))
    tok.save(os.path.join(run_dir, "tokenizer.json"))
    print(f"{cfg.run_name}: train {len(train_docs)}, held-out {len(val_docs)}, generated eval {len(eval_docs)} | {describe(device)}", flush=True)
    teacher_report, pseudo = None, None
    if cfg.kd_teacher:
        pseudo, teacher_report = _kd_targets(cfg, report, train_docs, manifest, device, run_dir)
    examples = []
    for i, doc in enumerate(train_docs, 1):
        gold = data.prepare(doc, tok, cfg, mtype)
        teacher = data.prepare(doc, tok, cfg, mtype, pseudo[doc["id"]]) if pseudo else None
        examples.append((gold, teacher))
        if i % 1000 == 0:
            print(f"prepared {i}/{len(train_docs)} training docs", flush=True)
    # Teacher generation seeds do not influence student training randomness.
    random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total = math.ceil(len(examples) / cfg.grad_accum) * cfg.epochs
    warm = int(total * cfg.warmup_frac)
    def scale(s):
        return (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, scale)
    scaler = grad_scaler(device)
    best, steps, stopped = -float("inf"), 0, False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    references = {d["id"]: d["summary"] for d in eval_docs}
    last_eval_step = -1
    with open(os.path.join(run_dir, "log.csv"), "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "step", "train_loss", "val_nll", "selection_metric", "selection_score", "seconds"])
        def evaluate(epoch, loss):
            nonlocal best, last_eval_step
            model.eval()
            summed, tokens = 0.0, 0
            with torch.no_grad():
                for doc in eval_docs:
                    l, n = summary_loss(model, data.prepare(doc, tok, cfg, mtype), mtype, device)
                    summed += l.item(); tokens += n
            predictions = generate_docs(model, tok, mtype, eval_docs, cfg, device)
            metrics = score_summaries(predictions, references, cfg.selection_metric == "bertscore_f1",
                                      cfg.bert_model, cfg.bert_device)
            metrics["val_nll"] = summed / tokens
            metric = metrics[cfg.selection_metric]
            writer.writerow([epoch, steps, loss, metrics["val_nll"], cfg.selection_metric, metric, time.time() - start])
            log.flush()
            print(f"[val] step {steps} {cfg.selection_metric}={metric:.4f} nll={metrics['val_nll']:.4f}", flush=True)
            if metric > best:
                best = metric
                _save(model, cfg, run_dir, dict(epoch=epoch, step=steps, metrics=metrics))
                dump(predictions, os.path.join(run_dir, "val_predictions.json"))
                dump(metrics, os.path.join(run_dir, "val_metrics.json"))
            last_eval_step = steps
            model.train()
        for epoch in range(1, cfg.epochs + 1):
            order = list(range(len(examples)))
            random.Random(cfg.seed + epoch).shuffle(order)
            model.train()
            total_loss, total_tokens = 0.0, 0.0
            for group_start in range(0, len(order), cfg.grad_accum):
                if deadline and time.time() >= deadline:
                    stopped = True
                    break
                opt.zero_grad(set_to_none=True)
                denom, group_loss = 0.0, 0.0
                for i in order[group_start:group_start + cfg.grad_accum]:
                    gold, teacher = examples[i]
                    targets = [(gold, 1.0)] if teacher is None else [(gold, 1 - cfg.kd_alpha), (teacher, cfg.kd_alpha)]
                    for example, weight in targets:
                        if weight == 0:
                            continue
                        loss, ntokens = summary_loss(model, example, mtype, device)
                        scaler.scale(weight * loss).backward()
                        denom += weight * ntokens; group_loss += weight * loss.item()
                scaler.unscale_(opt)
                for p in params:
                    if p.grad is not None:
                        p.grad.div_(denom)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                old = scaler.get_scale()
                scaler.step(opt); scaler.update()
                if scaler.get_scale() >= old:
                    steps += 1; sched.step()
                    if hasattr(model, "post_step"):
                        model.post_step()
                opt.zero_grad(set_to_none=True)
                total_loss += group_loss; total_tokens += denom
                if steps == 1 or steps % 20 == 0:
                    print(f"epoch {epoch}/{cfg.epochs} step {steps}/{total} loss {total_loss / total_tokens:.4f}", flush=True)
                if cfg.eval_every and steps and steps % cfg.eval_every == 0 and steps != last_eval_step:
                    evaluate(epoch, total_loss / total_tokens)
            if last_eval_step != steps:
                evaluate(epoch, total_loss / max(total_tokens, 1))
            if stopped:
                break
    result = dict(run=cfg.run_name, phase1_run=cfg.phase1_run, method=cfg.method, kd_teacher=cfg.kd_teacher,
                  teacher_params=teacher_report, train_docs=len(train_docs), val_docs=len(val_docs),
                  eval_docs=len(eval_docs), selection_metric=cfg.selection_metric, best_score=best,
                  best_val_metrics=read(os.path.join(run_dir, "val_metrics.json")), optimizer_steps=steps,
                  total_seconds=time.time() - start, stopped_early=stopped,
                  peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None, **report)
    dump(result, os.path.join(run_dir, "results.json"))
    return result


def train_summ(cfg):
    try:
        return _train_summ(cfg)
    except BaseException as e:
        from ..rr.trainer import free_gpu_after_failure
        free_gpu_after_failure(e)
        raise


def predict_test(run_dir, data_dir=None, out_dir=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg, mtype, _ = load_summ_run(run_dir, device)
    docs = data.load_split(os.path.join(data_dir or cfg.data_dir, "test.json"), labeled=False)
    manifest = read(os.path.join(run_dir, "split.json"))
    known = set(manifest["train_pool_ids"]) | set(manifest["val_ids"])
    if known.intersection(d["id"] for d in docs):
        raise ValueError("Test IDs overlap the public training/validation IDs")
    start = time.time()
    predictions = generate_docs(model, tok, mtype, docs, cfg, device)
    if any(not isinstance(s, str) or not s.strip() for s in predictions.values()):
        raise ValueError("Empty summary generated; inspect model/decoding before submitting")
    records = [dict(id=d["id"], summary=predictions[d["id"]]) for d in docs]
    out_dir = out_dir or os.path.join(run_dir, "submission")
    path = write_json_submission(records, {d["id"] for d in docs}, out_dir)
    dump(dict(n_test=len(docs), inference_seconds=time.time() - start,
              mean_words=sum(len(s.split()) for s in predictions.values()) / len(docs),
              decoding={k: getattr(cfg, k) for k in ("max_new_tokens", "min_new_tokens", "temperature", "top_k", "top_p", "repetition_penalty", "no_repeat_ngram_size")}),
         os.path.join(out_dir, "inference.json"))
    return path, records
