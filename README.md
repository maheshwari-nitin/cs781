# CS781 capstone: in-house legal intelligence

**Nitin Maheshwari · 230711 · IIT Kanpur**  
Codabench account: `n_230711` · Contact: `nitinm23@iitk.ac.in`

This project trains compact legal language models from scratch and adapts them to seven task interfaces.
The repository contains four foundation architectures, shared training/data utilities, task-specific
adaptation pipelines, notebooks, and regression tests. It also contains the LaTeX report and
an auditable transcription of the supplied competition results.

GitHub repository: [maheshwari-nitin/cs781](https://github.com/maheshwari-nitin/cs781).
No model weights, dataset files, access tokens or notebook outputs are included in the submission
package. The report's LLM declaration documents assistance explicitly.

## Results and evidence

The following are **best Finished competition uploads**, transcribed from `cs781_results.docx`.
These are not newly reproduced test scores. Scores across different tasks should not be averaged.

| Phase | Task | Primary metric | Best score | Reported rank | Scored uploads |
|---|---|---|---:|---:|---:|
| 1 | Legal LM continuation | BERTScore F1 | 0.8273 | 106 | 5 |
| 2 | Legal NER | Strict macro-F1 | 0.3703 | 113 | 4 |
| 3 | Rhetorical roles | Macro-F1 | 0.5146 | 50 | 5 |
| 4 | Judgment prediction | Macro-F1 | 0.7935 | 22 | 4 |
| 5 | Statute identification | Macro-F1 | 0.3517 | 92 | 4 |
| 6 | Legal summarization | BERTScore F1 | 0.8153 | 119 | 4 |
| 7 | Prior-case retrieval | Micro-F1@7 | 0.1855 | 126 | 3 |

Phase 4 has four scored submissions, including the KD attempt. The individual statuses remain
in `report/evidence/official_submissions.csv` and the per-submission documentation. The best-score
column uses the best Finished upload for each phase.

## Repository organization

The working source uses:

```text
models/                 A/B/C/D foundation implementations; tiny smoke-test models
adapters/               Existing LoRA and LoRMA projection wrappers
lm_lib/                 Shared pretraining, data, generation and checkpoint infrastructure
lm_lib/ner/             Exact-offset NER
lm_lib/rr/              Sentence rhetorical roles
lm_lib/cjpe/            Binary judgment prediction
lm_lib/lsi/             Multi-label statute identification
lm_lib/summ/            Summary-only training, generation and evaluation
lm_lib/pcr/             Retrieval, pool checks, batching, KD and optional tuning recipes
notebooks/              Notebook entry points for all phases
tests/                  Regression suite
report/                 Report.tex, appendix, declaration, bibliography, vector figures and evidence
```

The final submission is reorganized for review:

```text
code/
  phase1/ ... phase7/    Best-family source snapshot, recipe, provenance, selected upload and launcher
  common/               Shared lm_lib/, models/, adapters/, clean notebooks and tests
  submissions/          Individual documentation for every displayed attempt
  evidence/             Numerical result transcriptions and implementation diagnostics
  requirements.txt
report/
  CS781-CP-Nitin-Maheshwari-230711-report.pdf
  Report.tex, appendix.tex, llm_declaration.tex, references.bib
  figures/, evidence/, support/
```

The single submission ZIP contains `code/` and `report/` directly at its root. A separate outer directory
is not added. The compiled report uses the course title format and eight required numbered sections;
its main body is limited to ten pages, with the declaration, references and appendix afterwards.

## Foundation architectures

All models expose `MODEL_TYPE` and `build_model(vocab_size, max_seq_len, **extra)`.
Decoder models return vocabulary logits and optional loss; B accepts source/decoder inputs. Downstream
feature extraction can use explicit `hidden_states`, B's `encode`, or the features immediately before
the tied language-model head. This avoids a large vocabulary projection for classification/retrieval.

| ID | Implementation | Structure | Original parameters |
|---|---|---|---:|
| A | `A_qwen_dense.py` | 22 layers, width 1024, 16 query/4 KV heads, QK norm, RoPE, RMSNorm, SwiGLU 2816 | 280,805,120 |
| B | `B_encdec.py` | 10 encoder + 10 decoder layers, width 1024, shared embedding, cross-attention | 284,483,328 |
| C | `C_gdn_hybrid.py` | 24 layers, 3 gated DeltaNet : 1 gated attention, width 1024, FFN 1920 | 291,581,376 |
| D | `D_mla_moe.py` | 16 layers, width 768, MLA, shared/routed experts, top-2 routing, optional MTP | 233,973,504 |

Parameter tying is counted once. A's exact count is independently recoverable from its projections,
normalizations and 32,000-token embedding. B/C counts are logged in the retained outputs; D's count is
derived from its logged classifier total minus the documented head size.

These are compact implementations inspired by the cited families, not downloaded Qwen/T5/DeepSeek
weights. See `models/README.md` and the report bibliography for sources. C needs FLA/CUDA kernels.

## Environment and authentication

Use Python 3.12 and a CUDA-compatible PyTorch build. Install `code/requirements.txt` in the extracted
submission tree. The working repository has the same dependencies: NumPy, tokenizers, ijson,
Hugging Face Hub, pandas, scikit-learn, rouge-score, bert-score, sacrebleu and nbformat. C additionally
needs `flash-linear-attention`; A QLoRA needs `bitsandbytes` and CUDA.

Only free Kaggle compute was used for training. Under that constraint, only A completed the
planned full pretraining run; the other foundations used partially pretrained checkpoints.
Historical dependency versions were not fully retained. The local BERTScore output records
`roberta-large_L17_no-idf_version=0.3.12(hug_trans=5.22.0)`.

Authenticate to the official gated dataset with `HF_TOKEN` in the environment or Colab Secrets.
Never place a token in a published repository or submission archive. The clean upload helper packages
source only. Colab notebooks locate a folder with the relevant modules or a fresh `cs781_code.zip`;
upload `lm_lib/`, `models/` and `adapters/` from `code/common/` or create that safe source ZIP.

```powershell
python code/common/make_code_zip.py
```

This writes `code/common/cs781_code.zip` for the notebook bootstrap. Use this distributed helper;
it has an explicit source-only allowlist and never adds a credential file.

## Phase 1: pretraining and continuation

Notebooks: `00_data_tokenizer.ipynb`, `00b_full_data.ipynb`, `01_train.ipynb`,
`02_generate_submit.ipynb`, `03_resume.ipynb`, `04_decoding_sweep.ipynb`.

The preparation recipe samples 10% of documents from each of Acts, High Courts, Supreme Court and
Tribunals, with a 0.5% held-out fraction of the selected documents. A 32k byte-level BPE is trained on
the sample. Token files use uint16. The full-data notebook preserves the validation documents.
Per-source counts and actual consumed-token logs are not present in this evidence snapshot.

Decoder windows have 1024 tokens. B uses a 768-token prefix and 256-token continuation, so only target
positions are directly scored. AdamW uses the warmup/stable/decay recipe, calibration chooses a physical
microbatch, and resumable checkpoints preserve optimizer/RNG state. Generation targets exactly 128 words
and supports temperature/top-k/top-p/repetition controls. The archived generator recipe is greedy with
repetition penalty 1.1 and a 256-new-token cap; the exact winning export was not retained.

Selected upload: **965311, `qwen_dense_best.zip`, 0.8273**. Other finished families are B 0.8016,
C 0.8144 and D 0.8077. A generic finished `submission.zip` scored 0.8114; its architecture is unknown.
The later full-data A checkpoint is not assumed to be the October 6 comparison checkpoint.

## Phase 2: legal NER

Entry notebook: `10_ner.ipynb`. Inputs are fold text/gold dictionaries and `labels.txt`.
Gold spans use exclusive ends. Raw document text is preserved; leading byte-BPE whitespace is trimmed
only in token offset records. BIO classification supports overlapping windows, deterministic owner
selection, optional mean overlap aggregation, argmax and constrained Viterbi decoding. Oracle alignment
and overlap diagnostics expose representability limits. The oracle's actual measured value was not saved.

Output: `{document_id: [[start, end, label, entity_text], ...]}`. The exact task scorer checks boundaries
and label identity. A is the selected family, **977453, `A_ner.zip`, 0.3703**. B/C/D score 0.0006,
0.2984 and 0.2918. The exact winning A optimizer/decoding config is not exported locally; the phase
recipe is explicitly reconstructed from archived defaults.

## Phase 3: rhetorical roles

Entry notebook: `20_rr.ipynb`. CL/IT examples provide sentences and labels. Sentence token ranges are
recorded in a concatenated stream; document context reads neighboring sentences through windows, while
an alternative sentence-only mode is implemented. A head predicts one of 13 roles per original sentence.
No CRF is used. Synthetic-data utilities in the original workspace were **not used for training**, as
confirmed by the student, and are excluded from the prepared best-code package.

Output: `{document_id: [one_role_per_sentence]}`. Selected upload: **992390, `A_rr_2.zip`, 0.5146**.
The previous A upload scored 0.4837; B/C/D scored 0.3410/0.4092/0.3891. The precise change between A
uploads is unrecorded, so the gain is not attributed to a specific augmentation or hyperparameter.

## Phase 4: court judgment prediction

Entry notebook: `30_cjpe.ipynb`. A document classifier pools token states per chunk and then chunk
vectors. It supports last/first-last/spread selection and mean/max/learned chunk aggregation. The retained
full-A run uses last four 512-token chunks, mean aggregation and threshold 0.5. Binary cross-entropy is
implemented as two-class cross-entropy; optional balanced weights use training class counts.

Training has 37,387 examples (22,067 label 0; 15,320 label 1), dev 1,988 and test 1,517.
Output: `[{'id': ..., 'pred': 0_or_1}, ...]`.
Selected finished upload: **1006421, `A_cj.zip`, 0.7935**. Local A macro-F1 is 0.814897.
A LoRMA and B finished uploads score 0.5634 and 0.5914. B KD reaches local dev 0.640701, but its
competition upload is marked Failed. That displayed score is not used as a valid competition result.

## Phase 5: legal statute identification

Entry notebook: `40_lsi.ipynb`. The classifier has one independent sigmoid per statute, with BCE-with-logits
loss. Gold IDs are deduplicated and mapped explicitly to persisted columns. Test labels are not read.
Optional training-only positive weights and global/per-label dev thresholds are available; the recorded
comparison uses a fixed 0.5 threshold and unweighted loss. Dev-tuned thresholds would be optimistic if
evaluated on that same dev set. D KD uses independent Bernoulli targets rather than a categorical softmax.

Recorded sizes: 42,750 train / 10,181 dev / 13,019 test; 100 observed statute IDs. The representation
uses four spread 512-token chunks. Output: `[{'id': ..., 'pred': [unique_integer_ids]}, ...]`.
Selected upload: **1016369, `A_lsi.zip`, 0.3517**. Local A macro/micro-F1 and subset accuracy are
0.345304/0.492551/0.093311. Other official families: B 0.0328, C 0.0951, D KD 0.0730.

## Phase 6: abstractive summarization

Entry notebook: `50_summ.ipynb`. The schema uses `document` and `summary`, not classification's `text`.
There is no public dev file. Missing/blank training summaries 299 and 4799 are skipped and reported;
7,028 usable documents split into 6,678 train and 350 holdout. Thirty-two holdout IDs are generated for
selection. Duplicate normalized source documents stay in one split.

Decoder training masks prompt labels and scores summary tokens plus EOT. B trains both encoder and
decoder. Source/target budgets are 640/256 tokens; target truncation and full-reference evaluation are
documented. Generation permits EOT after 16 tokens and caps at 256; the recorded runs use greedy
decoding with repetition controls off. RoBERTa-large is an evaluation-only BERTScore encoder.

Output: `[{'id': ..., 'summary': '...'}, ...]`. Selected family/run: **A LoRA**, upload
**1028885, `A6.zip`, 0.8153**. Local BERTScore/ROUGE-1/ROUGE-L are 0.826998/0.301680/0.197140.
A QLoRA scores 0.8094 officially, compared with 0.8153 for ordinary LoRA.
B LoRMA and C LoRA score 0.7651 and 0.7680. No completed summary KD run appears in the evidence.
Repeated text in saved examples is documented despite the relatively high BERTScore.

## Phase 7: prior-case retrieval

Entry notebooks: `60_pcr.ipynb`, `61_pcr_a_tuning.ipynb`.
Files are separate train/dev/test queries **and their corresponding candidate pools**. Official records
are ID-keyed, with sentence-list `text`; labeled queries contain `relevant_candidates`. IDs remain strings.
The shared Phase-1 feature encoder pools tokens/chunks, normalizes case vectors and ranks by cosine
similarity. The recorded run uses first/last two 256-token chunks. Training samples up to two positives
and four random negatives, excluding every known positive from the negative set.

Train: 827 queries / 4,320 candidates, with ten zero-positive queries skipped. Dev: 118 / 1,023.
Test: 237 / 1,727. Dev gold has 973 relevant pairs. All dev/test search uses the full corresponding pool.
Output: `{query_id: [exactly_7_unique_ranked_candidate_ids]}`. Micro-F1 compares against the complete gold
sets, including queries with more than seven positives.

Selected upload: **1038509, `A_pcr.zip`, 0.1855**. Local A micro-F1/P/R are
0.192329/0.209443/0.177801, or 173 hits among 826 outputs and 973 gold pairs. Local C/B micro-F1 is
0.081156/0.072262; official C/B is 0.0732/0.0549. Three finished retrieval uploads and no completed KD
row are visible. Candidate-pool membership checks do not exclude a query's own ID; the report records
observed self-matches and identifies this as a concrete ablation to test.

Physical query and equal-length chunk batches support gradient accumulation without adding in-batch
negatives. Automatic batch sizing preserves the selected effective query batch. Embedding retries
reduce the batch and retry the same chunks without dropping candidates.

The longer A recipe (30 epochs, LR 3e-5, refreshed negatives and early stopping), TF-IDF hard negatives
and larger-context variants are **unexecuted proposals** in this evidence snapshot. The comparison helper
keeps the original measured baseline on ties/regressions and rejects mismatched train/dev subsets.
Do not present their settings or code availability as an achieved improvement.

## Run from the submission tree

```powershell
python code/phase5/run.py show-config
python code/phase5/run.py train --config code/phase5/recipe.json --phase1-run /path/to/A_qwen_dense_full --data-dir /path/to/p5
python code/phase5/run.py predict --run-dir /path/to/trained/lsi_run
```

Each phase directory explains its files and command options. `recipe.json` is a **reconstructed recipe**
when the exact winning run config was not exported. `config_provenance.json` identifies the uncertainty.
Use the original external run config and tokenizer/checkpoint to reproduce a specific submission.
The code archive intentionally omits weights; inference requires the student's external trained weights.

To rerun CPU tests from the extracted tree:

```powershell
cd code/common
python -m unittest discover -s tests -v
```

Thirteen retained tests cover data schemas, losses, alignment of KD inputs, checkpoint loading, pool
isolation, metric denominators, batching gradients, OOM retries, sampling and baseline preservation.
GPU-only FLA/NF4 and actual capacity/performance need Colab preflights. The new padding diagnostic under
`report/support/diagnose_padding.py` uses tiny random models and is not a competition evaluation.

## Known limitations and interpretation

A is the strongest measured family, but pretraining is not matched: later checkpoint metadata records
16,800 updates for A, 600 for B and 200 for C. B scores only the final 256 positions of its 1024-token
pretraining window. Its downstream bidirectional feature path also lacks attention padding masks when
inputs are padded. These issues warrant controlled checks; they do not establish that encoder–decoder
models are intrinsically bad. A's bounded long-document context remains a limitation.

No raw early-phase error sets, complete class histograms, every original winning config, actual Phase-1
consumed-token log, is invented. The report separates actual screenshot scores,
local outputs, source defaults and planned work. The source contains optional methods not represented
by completed experiments. The LLM declaration lists code, tuning and report assistance rather than
claiming the entire implementation was unaided.

## References

Primary sources and exact bibliography are in `report/references.bib`: IL-TUR, Qwen3, T5, gated DeltaNet,
DeepSeek-V3, GQA, RoPE, RMSNorm, GLU variants, LoRA, LoRMA, QLoRA, BERTScore, ROUGE, BLEU and knowledge
distillation. PyTorch CUDA, bitsandbytes and scikit-learn documentation support the implementation notes.
The report uses the supplied `Report_LLM.zip` template and retains all eight mandatory numbered sections.
