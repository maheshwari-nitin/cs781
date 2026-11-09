# Phase 1: Legal LM pretraining

Selected competition upload: **qwen_dense_best.zip**, ID **965311**, primary score **0.8273** (Finished).
The selected family is A, the 280,805,120-parameter dense decoder. `best_model.py` is an unchanged
source snapshot; runtime imports the identical implementation in `../common/models/A_qwen_dense.py`.

## Configuration provenance

Exact winning decoding settings, tokenizer artifact hash and consumed-token count were not retained.

`recipe.json` is deliberately named as a reconstruction. It contains usable starting paths/settings
for the implementation; it does not certify exact reproduction of the leaderboard number. Replace it
with the original external run config when available. No checkpoint or raw dataset is distributed.

## Run

Install `../requirements.txt` in Python 3.12, and prepare the official files with the clean notebooks:
`../common/notebooks/00_data_tokenizer.ipynb`, `../common/notebooks/00b_full_data.ipynb`, `../common/notebooks/01_train.ipynb`, `../common/notebooks/02_generate_submit.ipynb`, `../common/notebooks/03_resume.ipynb`, `../common/notebooks/04_decoding_sweep.ipynb`.

```powershell
python code/phase1/run.py show-config
python code/phase1/run.py train --data-dir /path/to/prepared/phase1
python code/phase1/run.py predict --run-dir /path/to/trained/run
```

Commands are run from the extracted archive root. `--config`, `--out-dir`, and `--phase1-ckpt` allow
explicit overrides. Use the original Phase-1 checkpoint for all adaptations; A full-data pretraining
is distinct from the original Phase-1 10% architecture-comparison run. Legacy predictors use the
data path saved in their external run config. See the root code README for schemas and methodological details.

For Phase-1 prediction also pass `--data-dir /path/to/data` containing `test.json`. The legacy GPU generator expects bf16-capable hardware.
