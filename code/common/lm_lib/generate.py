"""Batched continuation generation for decoder-only and encoder-decoder models.

Model hooks used if present (otherwise a cache-free fallback re-runs the full sequence each step):
  decoder: model.generate_tokens(idx, max_new_tokens, sample_fn) -> LongTensor [B, n_new]
  encdec : model.encode(src) -> memory ; model.decode(memory, dec_in) -> logits [B, L, V]

Prompts in a batch are sorted by length and left-truncated to the batch minimum, so no padding
or attention masks are needed (a handful of the earliest prompt tokens may be dropped).
"""
import contextlib
import time

import torch

from .utils import forward


def sample_next(logits, temperature=0.0, top_k=0, top_p=1.0):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, -1)
        remove = probs.cumsum(-1) - probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1)


def apply_repetition_penalty(logits, context, penalty):
    if penalty == 1.0:
        return logits
    scores = logits.gather(1, context)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits.scatter(1, context, scores)


def _words_done(tok, new_ids, eot_id, n_words):
    done = []
    for row in new_ids.tolist():
        if eot_id in row:
            done.append(True)
            continue
        done.append(len(tok.decode(row).split()) > n_words)  # > so the n-th word is complete
    return all(done)


@torch.inference_mode()
def _generate_batch(model, model_type, idx, max_new, sampling, tok, eot_id, bos_id, n_words, seq_len, src_len,
                    banned=()):
    temperature, top_k, top_p, rep = (sampling.get("temperature", 0.0), sampling.get("top_k", 0),
                                      sampling.get("top_p", 1.0), sampling.get("repetition_penalty", 1.0))
    banned = list(banned)

    def pick(logits, context):
        logits = apply_repetition_penalty(logits.float(), context, rep)
        if banned:
            logits[:, banned] = float("-inf")
        return sample_next(logits, temperature, top_k, top_p)

    B = idx.size(0)
    if model_type == "decoder" and hasattr(model, "generate_tokens"):
        return model.generate_tokens(idx, max_new, lambda lg, ctx: pick(lg, ctx))

    if model_type == "decoder":
        seq = idx
        for t in range(max_new):
            ctx = seq[:, -seq_len:]
            logits, _, _ = forward(model, (ctx,))
            nxt = pick(logits[:, -1], seq)
            seq = torch.cat([seq, nxt[:, None]], 1)
            if (t + 1) % 16 == 0 and _words_done(tok, seq[:, idx.size(1):], eot_id, n_words):
                break
        return seq[:, idx.size(1):]

    src = idx[:, -src_len:]
    dec = torch.full((B, 1), bos_id, dtype=torch.long, device=idx.device)
    memory = model.encode(src) if hasattr(model, "encode") else None
    for t in range(max_new):
        logits = model.decode(memory, dec) if memory is not None else forward(model, (src, dec))[0]
        nxt = pick(logits[:, -1], torch.cat([src, dec], 1))
        dec = torch.cat([dec, nxt[:, None]], 1)
        if (t + 1) % 16 == 0 and _words_done(tok, dec[:, 1:], eot_id, n_words):
            break
    return dec[:, 1:]


LAST_STATS = {}   # length diagnostics of the most recent generate_continuations call


def generate_continuations(model, tok, model_type, prompts, seq_len, src_len=768, n_words=128,
                           max_new_tokens=256, batch_size=64, sampling=None, device="cuda", seed=0,
                           ban_special=True):
    """prompts: {id: text}. Returns {id: continuation text truncated to n_words words}.
    ban_special: never sample <|endoftext|> / <|pad|> / <|bos|>. The model learned to end documents with
    <|endoftext|>, but the task always wants exactly n_words words, so stopping early only loses words."""
    sampling = sampling or {}
    torch.manual_seed(seed)
    eot_id, bos_id = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|bos|>")
    banned = [i for i in (eot_id, tok.token_to_id("<|pad|>"), bos_id) if i is not None] if ban_special else []
    n_eot = 0
    if model_type == "decoder":
        max_prompt = seq_len - max_new_tokens
    else:
        max_prompt = src_len
        max_new_tokens = min(max_new_tokens, seq_len - src_len)
    ids = list(prompts)
    encs = tok.encode_batch([prompts[i] for i in ids])
    toks = {i: e.ids[-max_prompt:] for i, e in zip(ids, encs)}
    order = sorted(ids, key=lambda i: len(toks[i]), reverse=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()
    out, t0 = {}, time.time()
    model.eval()
    for b in range(0, len(order), batch_size):
        chunk = order[b:b + batch_size]
        L = min(len(toks[i]) for i in chunk)
        idx = torch.tensor([toks[i][-L:] for i in chunk], dtype=torch.long, device=device)
        with amp:
            new = _generate_batch(model, model_type, idx, max_new_tokens, sampling, tok, eot_id, bos_id,
                                  n_words, seq_len, src_len, banned)
        for i, row in zip(chunk, new.tolist()):
            if eot_id in row:
                row = row[:row.index(eot_id)]
                n_eot += len(tok.decode(row).split()) < n_words
            out[i] = " ".join(tok.decode(row).split()[:n_words])
        print(f"  generated {min(b + batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)", end="\r")
    short = sum(len(v.split()) < n_words for v in out.values())
    LAST_STATS.clear()
    LAST_STATS.update(n=len(out), short=short, short_eot=n_eot, short_token_cap=short - n_eot)
    print(f"\ndone in {time.time() - t0:.0f}s; {short}/{len(out)} continuations shorter than {n_words} words "
          f"({n_eot} stopped at <|endoftext|>, {short - n_eot} hit the {max_new_tokens}-token cap)")
    return {i: out[i] for i in ids}
