import importlib
import inspect
import time

import torch

A100_BF16_PEAK_FLOPS = 312e12
PARAM_MIN, PARAM_MAX = 50_000_000, 300_000_000


def load_arch(arch):
    """Import models/<arch>.py. It must define MODEL_TYPE and build_model(vocab_size, max_seq_len, **kw)."""
    try:
        mod = importlib.import_module(f"models.{arch}")
    except ModuleNotFoundError as e:
        if e.name != f"models.{arch}":
            raise  # a dependency inside the model file is missing (e.g. fla), not the model itself
        import difflib, os
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        avail = sorted(f[:-3] for f in os.listdir(models_dir) if f.endswith(".py") and not f.startswith("__"))
        close = difflib.get_close_matches(arch, avail, n=1)
        raise ModuleNotFoundError(f"No model file models/{arch}.py. Available: {avail}"
                                  + (f". Did you mean '{close[0]}'?" if close else "")) from None
    mod = importlib.reload(mod)
    if getattr(mod, "MODEL_TYPE", None) not in ("decoder", "encdec"):
        raise ValueError(f"models/{arch}.py must set MODEL_TYPE = 'decoder' or 'encdec'")
    return mod


def build(arch_mod, vocab_size, max_seq_len, extra=None):
    kw = dict(extra or {})
    sig = inspect.signature(arch_mod.build_model)
    if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        kw = {k: v for k, v in kw.items() if k in sig.parameters}
    return arch_mod.build_model(vocab_size=vocab_size, max_seq_len=max_seq_len, **kw)


def count_params(model):
    """Unique parameters (tied weights counted once) -- this is the number the 50M-300M rule applies to."""
    return sum(p.numel() for p in model.parameters())


def check_param_budget(model, strict=True):
    n = count_params(model)
    ok = PARAM_MIN <= n <= PARAM_MAX
    msg = f"parameters: {n / 1e6:.2f}M ({'OK' if ok else 'OUTSIDE 50M-300M'})"
    if strict and not ok:
        raise ValueError(msg)
    print(msg)
    return n


def forward(model, batch):
    """Run the model on a batch tuple and return (logits, loss, extras dict)."""
    out = model(*batch)
    if isinstance(out, tuple):
        logits, loss = out[0], out[1]
        extras = out[2] if len(out) > 2 and isinstance(out[2], dict) else {}
        return logits, loss, extras
    raise TypeError("model forward must return (logits, loss) or (logits, loss, extras_dict)")


def flops_per_token(model, n_params, seq_len, src_len=None):
    """Training FLOPs per token. Uses model.flops_per_token(seq_len[, src_len]) if defined (do this for
    MoE: active params, attention cost; encdec: split src/tgt), otherwise the 6N approximation."""
    f = getattr(model, "flops_per_token", None)
    if not callable(f):
        return 6.0 * n_params
    if src_len is not None and "src_len" in inspect.signature(f).parameters:
        return f(seq_len, src_len=src_len)
    return f(seq_len)


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        now = time.time()
        dt, self.t = now - self.t, now
        return dt
