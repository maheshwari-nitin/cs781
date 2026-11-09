"""Checkpointing that survives Colab disconnects.

Each save is written to fast local disk first, then copied to Google Drive in a background
thread (so training doesn't stall on slow Drive I/O) and atomically renamed into place.

Files in run_dir:
  latest.pt   full resumable state (model + optimizer + step + RNG + config)  ~12 bytes/param
  best.pt     weights only, best validation loss                              ~4 bytes/param
  final.pt    weights only, end of training (+ config.json, tokenizer.json alongside)
  stable.pt   optional resumable copy at the end of the WSD stable phase (to continue on more data)
  latest.json pointer with step / time, written after latest.pt is in place
"""
import json
import os
import random
import shutil
import tempfile
import threading
import time

import numpy as np
import torch


def rng_state():
    s = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class CheckpointManager:
    def __init__(self, run_dir, local_tmp=None):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.local_tmp = local_tmp or os.path.join(tempfile.gettempdir(), "ckpt_tmp")
        os.makedirs(self.local_tmp, exist_ok=True)
        self._thread = None

    def path(self, name):
        return os.path.join(self.run_dir, f"{name}.pt")

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _publish(self, local_file, name, pointer=None):
        dst = self.path(name)
        tmp = dst + ".tmp"
        shutil.copyfile(local_file, tmp)
        os.replace(tmp, dst)
        os.remove(local_file)
        if pointer is not None:
            ptmp = os.path.join(self.run_dir, "latest.json.tmp")
            with open(ptmp, "w") as f:
                json.dump(pointer, f)
            os.replace(ptmp, os.path.join(self.run_dir, "latest.json"))

    def save(self, obj, name, pointer=None, background=True):
        self.wait()  # never more than one copy in flight
        t0 = time.time()
        local = os.path.join(self.local_tmp, f"{name}.pt")
        torch.save(obj, local)
        if background:
            self._thread = threading.Thread(target=self._publish, args=(local, name, pointer), daemon=True)
            self._thread.start()
        else:
            self._publish(local, name, pointer)
        return time.time() - t0

    def save_resumable(self, model, optimizer, step, tokens_seen, best_val, cfg_dict, name="latest", extra=None):
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                     tokens_seen=tokens_seen, best_val=best_val, config=cfg_dict, rng=rng_state(),
                     extra=extra or {}, saved_at=time.time())
        pointer = dict(step=step, tokens_seen=tokens_seen, best_val=best_val, time=time.ctime()) if name == "latest" else None
        return self.save(state, name, pointer)

    def save_weights(self, model, name, cfg_dict, meta=None, background=True):
        state = dict(model=model.state_dict(), config=cfg_dict, meta=meta or {})
        return self.save(state, name, background=background)

    def load(self, name="latest", map_location="cpu"):
        p = self.path(name)
        if not os.path.exists(p):
            return None
        return torch.load(p, map_location=map_location, weights_only=False)
