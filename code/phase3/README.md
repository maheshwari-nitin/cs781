# Phase 3: Rhetorical-role prediction

Selected competition upload: **A_rr_2.zip**, ID **992390**, primary score **0.5146** (Finished).
The selected family is A, the 280,805,120-parameter dense decoder. `best_model.py` is an unchanged
source snapshot; runtime imports the identical implementation in `../common/models/A_qwen_dense.py`.

## Configuration provenance

A_rr_2.zip is the best upload, but the change from A_rr.zip is not retained. This recipe is an implementation reconstruction, not the exact winner config.

`recipe.json` is deliberately named as a reconstruction. It contains usable starting paths/settings
for the implementation; it does not certify exact reproduction of the leaderboard number. Replace it
with the original external run config when available. No checkpoint or raw dataset is distributed.

## Run

Install `../requirements.txt` in Python 3.12, and prepare the official files with the clean notebooks:
`../common/notebooks/20_rr.ipynb`.

```powershell
python code/phase3/run.py show-config
python code/phase3/run.py train --data-dir /path/to/prepared/phase3 --phase1-run /path/to/A_qwen_dense_full
python code/phase3/run.py predict --run-dir /path/to/trained/run
```

Commands are run from the extracted archive root. `--config`, `--out-dir`, and `--phase1-ckpt` allow
explicit overrides. Use the original Phase-1 checkpoint for all adaptations; A full-data pretraining
is distinct from the original Phase-1 10% architecture-comparison run. Legacy predictors use the
data path saved in their external run config. See the root code README for schemas and methodological details.
