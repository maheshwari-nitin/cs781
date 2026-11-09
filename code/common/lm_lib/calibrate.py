"""Find the largest micro-batch that fits in GPU memory for a given model + sequence length.

Runs real forward/backward/optimizer steps (bf16 autocast, AdamW states allocated) and
binary-searches the micro-batch size against peak *reserved* memory, which is what actually
OOMs. Calibration runs eagerly by default; torch.compile usually lowers activation memory,
so the result is slightly conservative.
"""
import gc

import torch

from .utils import forward


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def calibrate_micro_batch(model, optimizer, make_batch, mem_frac=0.90, max_mb=1024, dtype=torch.bfloat16,
                          verbose=True):
    """model/optimizer: throwaway instances on the GPU (weights get modified -- rebuild afterwards).
    make_batch(mb) -> batch tuple on the GPU. Returns (best_mb, {mb: peak_GB or None})."""
    if not torch.cuda.is_available():
        print("No GPU: skipping calibration, using micro_batch=2")
        return 2, {}
    total = torch.cuda.get_device_properties(0).total_memory
    budget = mem_frac * total
    results = {}

    def trial(mb):
        batch = None
        try:
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):  # 2nd iteration includes allocated optimizer state
                batch = make_batch(mb)
                with torch.autocast("cuda", dtype=dtype):
                    _, loss, _ = forward(model, batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del loss
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            ok = peak <= budget
            results[mb] = round(peak / 1e9, 2)
        except torch.cuda.OutOfMemoryError:
            ok, results[mb] = False, None
        finally:
            del batch
            optimizer.zero_grad(set_to_none=True)
            _cleanup()
        if verbose:
            print(f"  micro_batch={mb:4d}: {'fits' if ok else 'too big'} "
                  f"(peak {results[mb]} GB / budget {budget / 1e9:.1f} GB)")
        return ok

    model.train()
    lo, hi = 0, None
    mb = 1
    while mb <= max_mb:
        if trial(mb):
            lo, mb = mb, mb * 2
        else:
            hi = mb
            break
    if lo == 0:
        raise RuntimeError("Even micro_batch=1 does not fit -- reduce seq_len or model size, "
                           "or add activation checkpointing")
    if hi is not None:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if trial(mid):
                lo = mid
            else:
                hi = mid
    if verbose:
        print(f"calibrated micro_batch = {lo}")
    return lo, results


def plan_accumulation(micro_batch, global_batch_seqs):
    accum = max(1, round(global_batch_seqs / micro_batch))
    return accum, micro_batch * accum
