"""Token classifier on top of a Phase-1 model, plus the plumbing for adaptation methods.

Features come from the Phase-1 backbone without touching its code:
  1. model.hidden_states(ids)  if you define it (e.g. to use an intermediate layer or concatenate layers)
  2. model.encode(ids)         encoder-decoder models: the bidirectional encoder output
  3. otherwise the input to the tied LM head (= output of the final norm), captured with a forward pre-hook
     that stops the forward pass before the 32k-way vocabulary projection is computed.

Adaptation methods: built-ins "full", "linear_probe", "partial" (pure parameter freezing). Anything else
(LoRMA, LoRA, adapters, prefix tuning, ...) is loaded from adapters/<name>.py, which you implement:
    def apply(backbone: nn.Module, **kwargs) -> None
        # insert your modules / wrap Linear layers in place and set requires_grad on what should train
The classification head is always trainable.
"""
import importlib

import torch
import torch.nn as nn


class _StopForward(Exception):
    pass


def find_lm_head(model):
    emb = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    for m in model.modules():
        if isinstance(m, nn.Linear) and id(m.weight) in emb:
            return m
    raise ValueError("could not find a tied LM head; define model.hidden_states(ids) instead")


def hidden_states(model, ids):
    if hasattr(model, "hidden_states"):
        return model.hidden_states(ids)
    if hasattr(model, "encode"):
        return model.encode(ids)
    head = find_lm_head(model)
    box = {}

    def pre_hook(_, inputs):
        box["h"] = inputs[0]
        raise _StopForward

    handle = head.register_forward_pre_hook(pre_hook)
    try:
        model(ids)
    except _StopForward:
        pass
    finally:
        handle.remove()
    return box["h"]


class TokenClassifier(nn.Module):
    def __init__(self, backbone, n_tags, head=None, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        with torch.no_grad():
            dev = next(backbone.parameters()).device
            was_training = backbone.training
            backbone.eval()
            self.hidden_dim = hidden_states(backbone, torch.zeros(1, 8, dtype=torch.long, device=dev)).size(-1)
            backbone.train(was_training)
        self.dropout = nn.Dropout(dropout)
        self.head = head if head is not None else nn.Linear(self.hidden_dim, n_tags)

    def forward(self, ids):
        return self.head(self.dropout(hidden_states(self.backbone, ids)))


# ---------------------------------------------------------------- adaptation methods
def feature_blocks(backbone):
    """The stack of blocks that produces the features (encoder for encoder-decoder models)."""
    for name in (("encoder",) if hasattr(backbone, "encode") else ()) + ("layers", "blocks", "h"):
        m = getattr(backbone, name, None)
        if isinstance(m, nn.ModuleList):
            return m
    raise ValueError("could not find the backbone's block list")


def enable_grad_checkpointing(backbone):
    """Recompute each feature block's activations in the backward pass instead of storing them (much less GPU
    memory, ~30% more compute). Only active when gradients are being computed."""
    import torch.utils.checkpoint as cp
    blocks = feature_blocks(backbone)
    for blk in blocks:
        if getattr(blk, "_ckpt", False):
            continue
        fwd = blk.forward

        def wrapped(*args, _fwd=fwd, **kwargs):
            if torch.is_grad_enabled():
                return cp.checkpoint(_fwd, *args, use_reentrant=False, **kwargs)
            return _fwd(*args, **kwargs)

        blk.forward = wrapped
        blk._ckpt = True
    return len(blocks)


def apply_method(clf, method, **kw):
    bb = clf.backbone
    if method == "full":
        for p in bb.parameters():
            p.requires_grad = True
    elif method == "linear_probe":
        for p in bb.parameters():
            p.requires_grad = False
    elif method == "partial":
        # train only the last `k` feature blocks (+ the final norms); everything else frozen
        k = kw.get("k", 4)
        for p in bb.parameters():
            p.requires_grad = False
        for blk in list(feature_blocks(bb))[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for name, mod in bb.named_children():
            if "norm" in name:
                for p in mod.parameters():
                    p.requires_grad = True
    else:
        mod = importlib.import_module(f"adapters.{method}")
        mod.apply(bb, **kw)
    for p in clf.head.parameters():
        p.requires_grad = True
    return param_report(clf)


def param_report(clf):
    total = sum(p.numel() for p in clf.parameters())
    trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    return dict(total_params=total, trainable_params=trainable, trainable_pct=100.0 * trainable / max(total, 1))
