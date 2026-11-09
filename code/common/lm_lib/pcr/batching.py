"""GPU sizing on disposable models, preserving the chosen effective query batch.

Memory accounting follows https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management
Probes measure peak reserved PyTorch memory plus pre-existing device use, leaving configurable headroom.
"""
import gc
import random
import time
import traceback
from dataclasses import replace

import torch

from ..amp import grad_scaler


def largest_fitting(candidates, trial):
    """Find the largest admissible size, using growth then binary search over sorted choices."""
    choices = sorted(set(candidates))
    if not choices or choices[0] < 1:
        raise ValueError("Batch choices must be positive")
    checked = {}
    def fits(i):
        if i not in checked:
            checked[i] = trial(choices[i])
        return checked[i]
    if not fits(0):
        raise RuntimeError("Even batch size 1 exceeds the memory budget. Reduce chunks/negatives, "
                           "enable gradient checkpointing, or use a parameter-efficient method.")
    lo, hi = 0, 1
    while hi < len(choices) and fits(hi):
        lo, hi = hi, min(2 * hi, len(choices))
    hi = min(hi, len(choices))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return choices[lo]


def training_choices(effective_queries, maximum):
    """Divisors permit exact accumulation, without rounding the optimizer's effective batch."""
    return [n for n in range(1, min(effective_queries, maximum) + 1) if effective_queries % n == 0]


def calibrate_batches(cfg, device=None, candidate_count=None):
    """Return a resolved config. Probes modify a throwaway model, never the eventual training weights.

    The returned auto_batch=False prevents duplicate probing when passed to train_pcr.
    CPU keeps the explicit settings and does not load a probe model.
    """
    from .trainer import build_retriever, embed_token_batch, ranking_batch_losses
    started_calibration = time.perf_counter()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    report = dict(device=device, target_memory_fraction=cfg.batch_mem_frac,
                  train_trials=[], embedding_trials=[])
    resolved = replace(cfg, auto_batch=False, batch_calibration=report)
    if device != "cuda":
        report.update(status="skipped_no_cuda", train_batch_size=cfg.train_batch_size,
                      train_chunk_batch_size=cfg.train_chunk_batch_size, embed_batch_size=cfg.embed_batch_size,
                      effective_batch_queries=cfg.train_batch_size * cfg.grad_accum, seconds=0.0)
        print("No CUDA: retaining explicit batch sizes; GPU calibration skipped", flush=True)
        return resolved
    effective = cfg.effective_batch_queries or cfg.train_batch_size * cfg.grad_accum
    choices = training_choices(effective, cfg.max_train_batch)
    if candidate_count is None:
        from . import data
        queries, pool = data.load_split(cfg.data_dir, "train")
        pairs, _ = data.sample_pairs(queries, pool, cfg.negatives, cfg.max_positives, cfg.seed)
        candidate_count = max(len(pair["candidate_ids"]) for pair in pairs)
        del queries, pool, pairs
    if candidate_count < 2:
        raise ValueError("Calibration needs positive and negative candidates")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    python_rng = random.getstate()
    model = opt = scaler = params = None
    gc.collect(); torch.cuda.empty_cache()
    try:
        model, tok, _ = build_retriever(resolved, device)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        # Allocate ALL trainable AdamW states, including routed experts that random probes might miss.
        for p in params:
            p.grad = torch.zeros_like(p)
        opt.step(); opt.zero_grad(set_to_none=True)
        scaler = grad_scaler(device)
        gc.collect(); torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        external = max(0, total - free - torch.cuda.memory_reserved())
        report.update(gpu=torch.cuda.get_device_name(), total_vram_bytes=total,
                      other_device_memory_bytes=external, effective_batch_queries=effective,
                      candidate_count=candidate_count, full_length_chunks=cfg.chunk_len,
                      chunks_per_case=cfg.max_chunks, optimizer_states_allocated=True)
        print(f"Calibrating {report['gpu']}: target {cfg.batch_mem_frac:.0%} of {total / 2**30:.1f} GiB; "
              f"effective query batch {effective}", flush=True)
        generator = torch.Generator().manual_seed(cfg.seed)
        vocab = tok.get_vocab_size()
        def token_cases(count):
            flat = torch.randint(3 if vocab > 3 else 0, vocab,
                                 (count * cfg.max_chunks, cfg.chunk_len), generator=generator).tolist()
            return [flat[i:i + cfg.max_chunks] for i in range(0, len(flat), cfg.max_chunks)]

        def measured_trial(size, training):
            nonlocal scaler
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            oom = False
            losses = output = None
            try:
                if training:
                    model.train()
                    parts = token_cases(size * (candidate_count + 1))
                    pairs, qtokens, ctokens, kd = [], {}, {}, {}
                    cursor = 0
                    npos = min(cfg.max_positives or candidate_count - 1, candidate_count - 1)
                    for i in range(size):
                        qid = f"probe_q{i}"
                        qtokens[qid] = parts[cursor]; cursor += 1
                        ids = []
                        for j in range(candidate_count):
                            ident = f"probe_c{i}_{j}"
                            ids.append(ident); ctokens[ident] = parts[cursor]; cursor += 1
                        pairs.append(dict(query_id=qid, candidate_ids=ids, n_positive=npos))
                        kd[qid] = dict(candidate_ids=ids, scores=[0.0] * candidate_count)
                    # One warm iteration and a second iteration with resident optimizer states.
                    for _ in range(2):
                        opt.zero_grad(set_to_none=True)
                        losses = ranking_batch_losses(model, pairs, qtokens, ctokens, resolved, device,
                                                       kd if cfg.kd_teacher else None, cfg.temperature)
                        scaler.scale(losses.mean()).backward()
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                        scaler.step(opt); scaler.update()
                        if hasattr(model, "post_step"):
                            model.post_step()
                        opt.zero_grad(set_to_none=True)
                        del losses
                else:
                    model.eval()
                    rows = torch.randint(3 if vocab > 3 else 0, vocab,
                                         (size, cfg.chunk_len), generator=generator).tolist()
                    with torch.no_grad():
                        for _ in range(2):
                            output = embed_token_batch(model, rows, resolved, device).cpu()
                            del output
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError as error:
                oom = True
                traceback.clear_frames(error.__traceback__)
                # A failed scaled step is discarded, including the scaler's partial unscale state.
                losses = output = None
                scaler = grad_scaler(device)
            finally:
                opt.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_reserved()
            allocated = torch.cuda.max_memory_allocated()
            elapsed = time.perf_counter() - started
            fits = not oom and peak + external <= cfg.batch_mem_frac * total
            record = dict(batch_size=size, fits=fits, oom=oom, peak_reserved_bytes=peak,
                          peak_allocated_bytes=allocated, estimated_device_peak_bytes=peak + external,
                          seconds=elapsed)
            report["train_trials" if training else "embedding_trials"].append(record)
            print(f"  {'train queries' if training else 'embedding chunks'} {size}: "
                  f"{'fits' if fits else 'over budget/OOM'}; peak {peak / 2**30:.2f} GiB", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            return fits

        selected_train = largest_fitting(choices, lambda n: measured_trial(n, True))
        resolved.train_batch_size = selected_train
        resolved.grad_accum = effective // selected_train
        # Probe inference while AdamW state is STILL resident, matching dev evaluation during training.
        resolved.embed_batch_size = largest_fitting(range(1, cfg.max_embed_batch + 1),
                                                   lambda n: measured_trial(n, False))
        report.update(status="calibrated", train_batch_size=resolved.train_batch_size,
                      train_chunk_batch_size=resolved.train_chunk_batch_size, grad_accum=resolved.grad_accum,
                      embed_batch_size=resolved.embed_batch_size,
                      hit_train_cap=selected_train == choices[-1],
                      hit_embedding_cap=resolved.embed_batch_size == cfg.max_embed_batch,
                      seconds=time.perf_counter() - started_calibration)
        print(f"Resolved: {resolved.train_batch_size} queries/microbatch x {resolved.grad_accum} accumulation "
              f"= {effective} queries/step; embedding batch {resolved.embed_batch_size}", flush=True)
    finally:
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        model = opt = scaler = params = p = None
        gc.collect(); torch.cuda.empty_cache()
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng); random.setstate(python_rng)
    return resolved
