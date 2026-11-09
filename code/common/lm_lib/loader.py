"""Deterministic, resumable batch loader over a flat uint16 token file.

The file is cut into non-overlapping windows; window order is a seeded permutation per epoch.
The batch for any optimizer step is a pure function of (seed, step), so resuming from a
checkpoint only needs the step number -- no loader state is saved.
"""
import numpy as np
import torch


class TokenWindows:
    def __init__(self, bin_path, window, seed=1337):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.window = window
        self.n_windows = len(self.data) // window
        if self.n_windows == 0:
            raise ValueError(f"{bin_path} has fewer than {window} tokens")
        self.seed = seed
        self._perm_epoch, self._perm = -1, None

    def _window_id(self, k):
        epoch, pos = divmod(k, self.n_windows)
        if epoch != self._perm_epoch:
            self._perm = np.random.default_rng(self.seed + epoch).permutation(self.n_windows)
            self._perm_epoch = epoch
        return int(self._perm[pos])

    def get(self, sample_ids):
        """sample_ids: global sample indices -> int64 array [B, window]."""
        out = np.empty((len(sample_ids), self.window), dtype=np.int64)
        for i, k in enumerate(sample_ids):
            w = self._window_id(k)
            out[i] = self.data[w * self.window:(w + 1) * self.window]
        return out


class BatchMaker:
    """Turns token windows into model inputs for decoder-only or encoder-decoder models."""

    def __init__(self, bin_path, model_type, seq_len, src_len=None, bos_id=None, seed=1337, device="cuda"):
        self.model_type = model_type
        self.seq_len, self.src_len, self.bos_id = seq_len, src_len, bos_id
        window = seq_len + 1 if model_type == "decoder" else seq_len
        self.windows = TokenWindows(bin_path, window, seed)
        self.device = device
        if model_type == "encdec":
            assert src_len is not None and 0 < src_len < seq_len and bos_id is not None

    def _to_device(self, *arrays):
        out = []
        for a in arrays:
            t = torch.from_numpy(a)
            if self.device != "cpu" and torch.cuda.is_available():
                t = t.pin_memory().to(self.device, non_blocking=True)
            out.append(t)
        return out

    def from_windows(self, w):
        if self.model_type == "decoder":
            return self._to_device(w[:, :-1], w[:, 1:])                     # (idx, targets)
        S = self.src_len
        src, tgt = w[:, :S], w[:, S:]
        dec_in = np.concatenate([np.full((len(w), 1), self.bos_id, dtype=np.int64), tgt[:, :-1]], axis=1)
        return self._to_device(src, dec_in, tgt)                           # (src, dec_in, targets)

    def micro_batch(self, step, micro_idx, micro_bs, accum):
        start = (step * accum + micro_idx) * micro_bs
        return self.from_windows(self.windows.get(range(start, start + micro_bs)))

    def random_batch(self, bs, rng):
        """Used for validation: random (seeded) windows."""
        ids = rng.integers(0, self.windows.n_windows, size=bs)
        w = np.stack([self.windows.data[i * self.windows.window:(i + 1) * self.windows.window] for i in ids])
        return self.from_windows(w.astype(np.int64))
