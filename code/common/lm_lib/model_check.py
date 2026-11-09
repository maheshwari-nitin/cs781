"""Pre-flight tests for a model in models/ (run before spending GPU hours on it).

Checks: interface/shapes, parameter budget, init loss ~ ln(V), causality (no future leakage),
gradient flow (every trainable param gets a finite grad), a short overfit test on one batch,
encoder-decoder wiring, and KV-cache consistency of optional generation hooks.
"""
import math

import torch

from .utils import build, count_params, flops_per_token, forward, load_arch, PARAM_MIN, PARAM_MAX


def _report(results, name, ok, detail=""):
    results[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def _batch(model_type, V, B, T, src_len, device, g):
    r = lambda *s: torch.randint(3, V, s, device=device, generator=g)
    if model_type == "decoder":
        x = r(B, T + 1)
        return (x[:, :-1], x[:, 1:])
    # same slicing as lm_lib.loader: src/tgt are non-contiguous views of one window
    w = r(B, T)
    src, tgt = w[:, :src_len], w[:, src_len:]
    dec_in = torch.cat([torch.full((B, 1), 2, device=device), tgt[:, :-1]], 1)
    return (src, dec_in, tgt)


def check_model(arch, vocab_size, seq_len=1024, src_len=768, extra=None, device=None, test_len=128,
                overfit_steps=40, strict_budget=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mod = load_arch(arch)
    mt = mod.MODEL_TYPE
    torch.manual_seed(0)
    model = build(mod, vocab_size, seq_len, extra).to(device).float()
    g = torch.Generator(device=device).manual_seed(0)
    res = {}
    T = min(test_len, seq_len)
    S = min(src_len, T // 2) if mt == "encdec" else src_len

    # 1. parameter budget
    n = count_params(model)
    _report(res, "param budget 50M-300M", PARAM_MIN <= n <= PARAM_MAX or not strict_budget, f"{n / 1e6:.2f}M")

    # 2. interface, shapes, init loss
    model.eval()
    batch = _batch(mt, vocab_size, 2, T, S, device, g)
    with torch.no_grad():
        logits, loss, extras = forward(model, batch)
        tgt = batch[-1]
        shape_ok = logits is not None and tuple(logits.shape) == (*tgt.shape, vocab_size)
        _report(res, "logits shape", shape_ok, f"{None if logits is None else tuple(logits.shape)}, "
                                              f"expected {(*tgt.shape, vocab_size)}")
        lnv = math.log(vocab_size)
        # models whose loss includes auxiliary terms (MTP, balance loss) report the plain CE as extras['ce_loss']
        ce = extras.get("ce_loss", loss) if extras else loss
        _report(res, "init loss ~ ln(V)", ce is not None and abs(float(ce) - lnv) < 1.5,
                f"{float(ce):.3f} vs ln V = {lnv:.3f}" + (f" (total incl. aux terms {loss.item():.3f})" if ce is not loss else "")
                + " (far off => check init / output scaling)")
        no_tgt = forward(model, batch[:-1])
        _report(res, "targets=None returns loss None", no_tgt[1] is None)

    # 3. causality: changing tokens at positions >= k must not change logits at positions < k
    with torch.no_grad():
        k = T // 2 if mt == "decoder" else (T - S) // 2
        b2 = [t.clone() for t in batch[:-1]]
        seq = b2[0] if mt == "decoder" else b2[1]
        seq[:, k:] = torch.randint(3, vocab_size, seq[:, k:].shape, device=device, generator=g)
        l1 = forward(model, batch[:-1])[0]
        l2 = forward(model, tuple(b2))[0]
        diff = (l1[:, :k] - l2[:, :k]).abs().max().item()
        _report(res, "causal (no future leakage)", diff < 1e-3, f"max |delta logits| before position {k} = {diff:.2e}")
        if mt == "encdec":
            b3 = [t.clone() for t in batch[:-1]]
            b3[0] = torch.randint(3, vocab_size, b3[0].shape, device=device, generator=g)
            d = (forward(model, tuple(b3))[0] - l1).abs().max().item()
            _report(res, "decoder depends on encoder (cross-attn wired)", d > 1e-4, f"max delta = {d:.2e}")

    # 4. gradient flow
    model.train()
    model.zero_grad(set_to_none=True)
    _, loss, _ = forward(model, batch)
    loss.backward()
    missing = [nme for nme, p in model.named_parameters() if p.requires_grad and p.grad is None]
    bad = [nme for nme, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    # MoE routed experts that received no tokens in this small batch legitimately get no gradient
    unrouted = sorted({nme.rsplit(".", 2)[0] for nme in missing if ".experts." in nme})
    missing = [nme for nme in missing if ".experts." not in nme]
    _report(res, "every param gets a grad", not missing, f"no grad: {missing[:8]}" if missing else "")
    if unrouted:
        print(f"[WARN] {len(unrouted)} routed expert(s) got no tokens in this {batch[0].numel()}-token batch "
              f"(expected for a fresh router; load balancing should fix it in training): {unrouted[:6]}")
    _report(res, "grads finite", not bad, f"non-finite: {bad[:8]}" if bad else "")
    model.zero_grad(set_to_none=True)

    # 5. overfit one batch
    if overfit_steps:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        small = _batch(mt, vocab_size, 2, T, S, device, g)
        first = None
        for i in range(overfit_steps):
            _, loss, _ = forward(model, small)
            loss.backward()
            opt.step(); opt.zero_grad(set_to_none=True)
            if hasattr(model, "post_step"):
                model.post_step()
            first = loss.item() if first is None else first
        _report(res, "overfits one batch", loss.item() < 0.5 * first, f"{first:.3f} -> {loss.item():.3f}")
        del opt

    # 6. generation hooks agree with the plain forward pass
    model.eval()
    with torch.no_grad():
        if mt == "decoder" and hasattr(model, "generate_tokens"):
            idx = batch[0][:, :T // 2]
            fast = model.generate_tokens(idx, 16, lambda lg, ctx: lg.argmax(-1))
            seq = idx
            for _ in range(16):
                seq = torch.cat([seq, forward(model, (seq,))[0][:, -1].argmax(-1, keepdim=True)], 1)
            _report(res, "generate_tokens == uncached greedy", torch.equal(fast, seq[:, idx.size(1):]))
        if mt == "encdec" and hasattr(model, "encode") and hasattr(model, "decode"):
            d = (model.decode(model.encode(batch[0]), batch[1]) - forward(model, batch[:-1])[0]).abs().max().item()
            _report(res, "decode(encode(src)) == forward", d < 1e-3, f"max delta = {d:.2e}")

    if hasattr(model, "flops_per_token"):
        fpt = flops_per_token(model, n, seq_len, src_len)
        print(f"flops_per_token(seq_len={seq_len}) = {fpt / 1e9:.3f} GFLOPs (6N = {6 * n / 1e9:.3f})")
    ok = all(res.values())
    print(f"\n{arch}: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res
