# Phase 5: Legal statute identification

Selected competition upload: **A_lsi.zip**, ID **1016369**, primary score **0.3517** (Finished).
The selected family is A, the 280,805,120-parameter dense decoder. `best_model.py` is an unchanged
source snapshot; runtime imports the identical implementation in `../common/models/A_qwen_dense.py`.

## Configuration provenance

Run A_full__spread4x512_3 and its scores are observed. The final saved configuration cell belongs to D; A epoch/checkpoint choices require the original run export.

`recipe.json` is deliberately named as a reconstruction. It contains usable starting paths/settings
for the implementation; it does not certify exact reproduction of the leaderboard number. Replace it
with the original external run config when available. No checkpoint or raw dataset is distributed.

## Run

Install `../requirements.txt` in Python 3.12, and prepare the official files with the clean notebooks:
`../common/notebooks/40_lsi.ipynb`.

```powershell
python code/phase5/run.py show-config
python code/phase5/run.py train --data-dir /path/to/prepared/phase5 --phase1-run /path/to/A_qwen_dense_full
python code/phase5/run.py predict --run-dir /path/to/trained/run
```

Commands are run from the extracted archive root. `--config`, `--out-dir`, and `--phase1-ckpt` allow
explicit overrides. Use the original Phase-1 checkpoint for all adaptations; A full-data pretraining
is distinct from the original Phase-1 10% architecture-comparison run. Legacy predictors use the
data path saved in their external run config. See the root code README for schemas and methodological details.
