"""Training loop: WSD learning-rate schedule, gradient accumulation, bf16 autocast, periodic
validation, time-based checkpointing to Drive and automatic resume."""
import contextlib
import csv
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from .calibrate import calibrate_micro_batch, plan_accumulation
from .ckpt import CheckpointManager, set_rng_state
from .config import TrainConfig
from .loader import BatchMaker
from .utils import build, check_param_budget, count_params, flops_per_token, forward, load_arch, \
    A100_BF16_PEAK_FLOPS


def wsd_lr(step, total, cfg: TrainConfig):
    """Warmup-Stable-Decay: linear warmup, constant, then linear decay to min_lr over the last decay_frac."""
    warm = max(1, int(cfg.warmup_frac * total))
    decay_start = int((1 - cfg.decay_frac) * total)
    if step < warm:
        return cfg.lr * (step + 1) / warm
    if step < decay_start:
        return cfg.lr
    frac = (step - decay_start) / max(1, total - decay_start)
    return cfg.lr * (1 - (1 - cfg.min_lr_ratio) * min(1.0, frac))


def make_optimizer(model, cfg: TrainConfig, device):
    if hasattr(model, "configure_param_groups"):        # optional model hook
        groups = model.configure_param_groups(cfg.weight_decay)
    else:
        decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
        groups = [dict(params=decay, weight_decay=cfg.weight_decay), dict(params=no_decay, weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                             fused=(device == "cuda"))


def _autocast(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device == "cuda" else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, bm: BatchMaker, cfg: TrainConfig, mb, device, dtype):
    """Returns val loss over all predicted positions and, for decoder models, the 'tail' loss on
    positions >= src_len (the same tokens / context the encoder-decoder is scored on), so losses
    are comparable across architectures."""
    model.eval()
    rng = np.random.default_rng(0)      # identical val batches every time
    tot = tail = 0.0
    n_tail = 0
    for _ in range(cfg.eval_batches):
        batch = bm.random_batch(mb, rng)
        with _autocast(device, dtype):
            logits, loss, _ = forward(model, batch)
        tot += loss.float().item()
        if bm.model_type == "decoder" and logits is not None and logits.dim() == 3 \
                and logits.size(1) == batch[0].size(1):
            S = cfg.src_len
            tl = F.cross_entropy(logits[:, S:].float().reshape(-1, logits.size(-1)), batch[1][:, S:].reshape(-1))
            tail += tl.item(); n_tail += 1
    model.train()
    out = dict(val_loss=tot / cfg.eval_batches)
    if bm.model_type == "encdec":
        out["val_tail_loss"] = out["val_loss"]
    elif n_tail:
        out["val_tail_loss"] = tail / n_tail
    return out


def train(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg.dtype) if device == "cuda" else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from tokenizers import Tokenizer
    tok_path = os.path.join(cfg.data_dir, "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    vocab_size = tok.get_vocab_size()
    bos_id = tok.token_to_id("<|bos|>")

    arch_mod = load_arch(cfg.arch)
    model_type = arch_mod.MODEL_TYPE
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    ckpt = CheckpointManager(run_dir)
    resume = ckpt.load("latest")
    if resume is not None:
        saved = resume["extra"]
        print(f"Resuming {cfg.run_name} from step {resume['step']} (best val {resume['best_val']:.4f})")
        cfg.micro_batch = saved.get("micro_batch", cfg.micro_batch)

    def new_model(seed):
        torch.manual_seed(seed)
        return build(arch_mod, vocab_size, cfg.seq_len, cfg.extra).to(device)

    train_bm = BatchMaker(os.path.join(cfg.data_dir, cfg.train_bin), model_type, cfg.seq_len, cfg.src_len,
                          bos_id, cfg.seed, device)
    val_bm = BatchMaker(os.path.join(cfg.data_dir, "val.bin"), model_type, cfg.seq_len, cfg.src_len,
                        bos_id, cfg.seed, device)

    # ---- batch-size calibration (fresh throwaway model, then rebuilt) ----
    if cfg.micro_batch <= 0:
        m = new_model(cfg.seed)
        opt = make_optimizer(m, cfg, device)
        rng = np.random.default_rng(1)
        cfg.micro_batch, table = calibrate_micro_batch(
            m, opt, lambda b: train_bm.random_batch(b, rng), cfg.mem_frac, dtype=dtype)
        del m, opt
        if device == "cuda":
            torch.cuda.empty_cache()
    accum, global_seqs = plan_accumulation(cfg.micro_batch, cfg.global_batch_seqs)
    mb = cfg.micro_batch

    model = new_model(cfg.seed)
    n_params = check_param_budget(model, strict=(device == "cuda"))
    optimizer = make_optimizer(model, cfg, device)

    window = train_bm.windows.window
    tokens_per_step = global_seqs * window
    budget = cfg.train_tokens or train_bm.windows.n_windows * window
    total_steps = max(1, budget // tokens_per_step)
    stable_end = int((1 - cfg.decay_frac) * total_steps)
    fpt = flops_per_token(model, n_params, cfg.seq_len, cfg.src_len)
    print(f"{cfg.arch} ({model_type}) | micro_batch={mb} x accum={accum} = {global_seqs} seqs "
          f"= {tokens_per_step:,} tok/step | {total_steps} steps = {total_steps * tokens_per_step / 1e6:.0f}M tokens")

    step, tokens_seen, best_val = 0, 0, float("inf")
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        step, tokens_seen, best_val = resume["step"], resume["tokens_seen"], resume["best_val"]
        set_rng_state(resume["rng"])
        del resume
    elif cfg.init_from:
        src = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        src_arch = src.get("config", {}).get("arch")
        if src_arch and src_arch != cfg.arch:
            raise ValueError(f"init_from checkpoint is arch {src_arch}, but this run is {cfg.arch}")
        model.load_state_dict(src["model"])
        print(f"initialised weights from {cfg.init_from} (meta: {src.get('meta') or {'step': src.get('step')}}); "
              f"optimizer and LR schedule start fresh")
        del src

    raw_model = model
    if cfg.compile and device == "cuda":
        model = torch.compile(raw_model)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        f.write(cfg.to_json())
    shutil.copyfile(tok_path, os.path.join(run_dir, "tokenizer.json"))
    log_path = os.path.join(run_dir, "log.csv")
    log_f = open(log_path, "a", newline="")
    log_w = csv.writer(log_f)
    if os.path.getsize(log_path) == 0:
        log_w.writerow(["step", "tokens", "lr", "loss", "grad_norm", "tok_per_s", "mfu", "val_loss",
                        "val_tail_loss", "mem_gb", "extras"])

    extra_state = dict(micro_batch=mb, accum=accum, n_params=n_params, total_steps=total_steps)
    t_start = t_ckpt = t_log = time.time()
    tokens_at_log = tokens_seen
    model.train()
    status = "finished"

    while step < total_steps:
        lr = wsd_lr(step, total_steps, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr
        loss_acc = 0.0
        extras = {}
        for micro in range(accum):
            batch = train_bm.micro_batch(step, micro, mb, accum)
            with _autocast(device, dtype):
                _, loss, extras = forward(model, batch)
            (loss / accum).backward()
            loss_acc += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if hasattr(raw_model, "post_step"):
            raw_model.post_step()
        step += 1
        tokens_seen += tokens_per_step

        val = {}
        if step % cfg.eval_every == 0 or step == total_steps:
            val = evaluate(model, val_bm, cfg, mb, device, dtype)
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                ckpt.save_weights(raw_model, "best", json.loads(cfg.to_json()), dict(step=step, **val))
            print(f"  [eval] step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in val.items())
                  + f" (best {best_val:.4f})")

        if step % cfg.log_every == 0 or val:
            dt = time.time() - t_log
            tps = (tokens_seen - tokens_at_log) / max(dt, 1e-9)
            mfu = fpt * tps / A100_BF16_PEAK_FLOPS if device == "cuda" else 0.0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            l = (loss_acc / accum).item()
            ex = {k: round(float(v), 4) for k, v in (extras or {}).items()}
            print(f"step {step}/{total_steps} | loss {l:.4f} | lr {lr:.2e} | gnorm {float(gnorm):.2f} | "
                  f"{tps / 1e3:.1f}k tok/s | MFU {mfu * 100:.1f}% | mem {mem:.1f}GB" + (f" | {ex}" if ex else ""))
            log_w.writerow([step, tokens_seen, lr, round(l, 5), round(float(gnorm), 4), round(tps), round(mfu, 4),
                            val.get("val_loss", ""), val.get("val_tail_loss", ""), round(mem, 2),
                            json.dumps(ex) if ex else ""])
            log_f.flush()
            t_log, tokens_at_log = time.time(), tokens_seen

        if cfg.snapshot_every and step % cfg.snapshot_every == 0 and step < total_steps:
            ckpt.save_weights(raw_model, f"step_{step:06d}", json.loads(cfg.to_json()), dict(step=step, **val))
            print(f"  [snapshot] step_{step:06d}.pt")
        if cfg.keep_stable_ckpt and step == stable_end:
            ckpt.save_resumable(raw_model, optimizer, step, tokens_seen, best_val, json.loads(cfg.to_json()),
                                name="stable", extra=extra_state)
        if time.time() - t_ckpt > cfg.ckpt_minutes * 60:
            s = ckpt.save_resumable(raw_model, optimizer, step, tokens_seen, best_val, json.loads(cfg.to_json()),
                                    extra=extra_state)
            print(f"  [ckpt] latest @ step {step} ({s:.0f}s local write, Drive copy in background)")
            t_ckpt = time.time()
        if cfg.max_minutes and time.time() - t_start > cfg.max_minutes * 60 and step < total_steps:
            status = "paused"
            break

    ckpt.save_resumable(raw_model, optimizer, step, tokens_seen, best_val, json.loads(cfg.to_json()),
                        extra=extra_state)
    if status == "finished":
        ckpt.save_weights(raw_model, "final", json.loads(cfg.to_json()), dict(step=step, best_val=best_val))
    ckpt.wait()
    log_f.close()
    hours = (time.time() - t_start) / 3600
    print(f"{status}: step {step}/{total_steps}, {tokens_seen / 1e6:.0f}M tokens, best val {best_val:.4f}, "
          f"{hours:.2f}h this session -> {run_dir}")
    return dict(status=status, step=step, total_steps=total_steps, best_val=best_val, run_dir=run_dir)


def load_trained(run_dir, which="best", device="cuda"):
    """Rebuild a model from run_dir/{best,final,latest}.pt + config.json."""
    from tokenizers import Tokenizer
    state = torch.load(os.path.join(run_dir, f"{which}.pt"), map_location="cpu", weights_only=False)
    cfg = TrainConfig.from_dict(state["config"])
    tok = Tokenizer.from_file(os.path.join(run_dir, "tokenizer.json"))
    arch_mod = load_arch(cfg.arch)
    model = build(arch_mod, tok.get_vocab_size(), cfg.seq_len, cfg.extra)
    model.load_state_dict(state["model"])
    print(f"loaded {which}.pt from {run_dir}: {count_params(model) / 1e6:.1f}M params, meta={state.get('meta')}")
    return model.to(device).eval(), tok, cfg, arch_mod.MODEL_TYPE
