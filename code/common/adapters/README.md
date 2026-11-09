# Adaptation methods for Phases 2–7 (you implement these)

Built into `lm_lib/ner/model.py` (pure parameter freezing, no new modules):
`full` (full fine-tuning), `linear_probe` (frozen backbone, head only), `partial` (last `k` blocks + final norms;
`method_kwargs={"k": 4}`).

Anything else is a file `adapters/<name>.py` that defines:

```python
def apply(backbone: torch.nn.Module, **kwargs) -> None:
    """Modify the Phase-1 backbone in place: wrap/replace nn.Linear layers or insert modules,
    and set requires_grad=True only on the parameters that should train (freeze the rest)."""
```

Then train it with `NERConfig(method="<name>", method_kwargs={...})`. The trainer:
- builds the backbone from the Phase-1 checkpoint, calls `apply`, and moves everything to the GPU;
- freezes any trainable parameter that receives no gradient (for example, the decoder of B, whose encoder makes the features);
- counts trainable vs total parameters and saves them in `results.json` for the efficiency comparison.

Useful helpers: `lm_lib.ner.model.feature_blocks(backbone)` returns the block list that produces the features (the encoder
for B, `layers` for A/C/D). In the Phase-1 models the attention projections are `q_proj/k_proj/v_proj/o_proj`
(MLA in D: `q_down/q_up_*/kv_down/kv_up_*`) and the MLPs `gate/up/down`.

Required by the brief (§8.2): **LoRMA is compulsory** (Bihany, Patel & Modi, 2025), plus ≥3 different techniques and
1 knowledge-distillation run. KD is built into the trainer: train a teacher run first, then train a smaller student
with `kd_teacher=<teacher run dir>`, `kd_alpha`, `kd_temperature`.
