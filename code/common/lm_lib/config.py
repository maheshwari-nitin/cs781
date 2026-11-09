from dataclasses import dataclass, asdict, field
import json


@dataclass
class TrainConfig:
    # --- identity / paths ---
    arch: str = "A_qwen_dense"           # module name inside models/
    run_name: str = ""                   # defaults to arch
    data_dir: str = "/content/data"      # local copy of train.bin / val.bin / tokenizer.json
    out_dir: str = "/content/drive/MyDrive/cs781/runs"  # checkpoints go to out_dir/run_name
    train_bin: str = "train.bin"         # file in data_dir: train.bin (10% sample) or train_full.bin (all data)
    init_from: str = ""                  # optional .pt (best/final/latest of another run): start from its
                                         # weights (fresh optimizer + schedule); ignored when resuming this run

    # --- sequence / batch ---
    seq_len: int = 1024                  # decoder: tokens per window; encdec: src_len + tgt_len
    src_len: int = 768                   # encdec only: encoder tokens (rest of seq_len is the decoder target)
    global_batch_seqs: int = 256         # target sequences per optimizer step (~262k tokens at 1024)
    micro_batch: int = 0                 # 0 => auto-calibrate on the GPU
    mem_frac: float = 0.90               # calibration: max fraction of GPU memory to use

    # --- budget ---
    train_tokens: int = 0                # 0 => one pass over train.bin

    # --- optimisation (starting points from the literature; tune yourself) ---
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    decay_frac: float = 0.20             # WSD: last 20% of steps decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- system ---
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337

    # --- logging / eval / checkpoint ---
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 40
    ckpt_minutes: float = 15.0
    max_minutes: float = 0.0             # >0 => stop + checkpoint after this many minutes (Colab limits)
    keep_stable_ckpt: bool = False       # keep a resumable copy at the end of the WSD stable phase
    snapshot_every: int = 0              # >0 => also keep weights-only snapshots step_XXXXXX.pt every N steps
                                         # (for checkpoint comparison; ~4 bytes/param each)

    extra: dict = field(default_factory=dict)  # free-form, passed to build_model(**extra)

    def __post_init__(self):
        if not self.run_name:
            self.run_name = self.arch

    def to_json(self):
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)
