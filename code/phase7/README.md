# Phase 7: Prior-case retrieval

Selected competition upload: **A_pcr.zip**, ID **1038509**, primary score **0.1855** (Finished).
The selected family is A, the 280,805,120-parameter dense decoder. `best_model.py` is an unchanged
source snapshot; runtime imports the identical implementation in `../common/models/A_qwen_dense.py`.

## Configuration provenance

A full local run, physical/effective batch 32, embedding batch 1024 and 3-epoch source recipe are observed. The exact run artifact/config is external.

`recipe.json` is deliberately named as a reconstruction. It contains usable starting paths/settings
for the implementation; it does not certify exact reproduction of the leaderboard number. Replace it
with the original external run config when available. No checkpoint or raw dataset is distributed.

## Run

Install `../requirements.txt` in Python 3.12, and prepare the official files with the clean notebooks:
`../common/notebooks/60_pcr.ipynb`, `../common/notebooks/61_pcr_a_tuning.ipynb`.

```powershell
python code/phase7/run.py show-config
python code/phase7/run.py train --data-dir /path/to/prepared/phase7 --phase1-run /path/to/A_qwen_dense_full
python code/phase7/run.py predict --run-dir /path/to/trained/run
```

Commands are run from the extracted archive root. `--config`, `--out-dir`, and `--phase1-ckpt` allow
explicit overrides. Use the original Phase-1 checkpoint for all adaptations; A full-data pretraining
is distinct from the original Phase-1 10% architecture-comparison run. Legacy predictors use the
data path saved in their external run config. See the root code README for schemas and methodological details.
