# Report source and evidence

Author: Nitin Maheshwari, 230711, nitinm23@iitk.ac.in. Codabench: n_230711.

The final named PDF is `CS781-CP-Nitin-Maheshwari-230711-report.pdf`.
The **main report occupies pages 1–10**, including the title and abstract. The required LLM
declaration, references and appendix follow the main report.
The layout uses the supplied template's 11-point article and one-inch margins.
`audit.json` records the page boundary, word count, compiler warnings and page summaries.

## Rebuild the PDF

Run from the extracted archive root:

```powershell
tectonic --keep-logs --keep-intermediates report/Report.tex
```

Tectonic downloads its TeX support files on first use. Alternatively, use a standard TeX installation:

```powershell
latexmk -xelatex -cd report/Report.tex
```

Both commands produce `report/Report.pdf`; copy it to the required submission filename after checking
the main-body page count. The included figures and generated table sources are sufficient to compile
the report. No datasets, checkpoints, compiler executables or credentials are required.

## Evidence and figures

`evidence/official_submissions.json` and `.csv` transcribe all 33 displayed attempts from the student's
results document. Phase 4 has four scored submissions, including the KD attempt. Failed, Preparing and Cancelled attempts
remain visible in the appendix ledger and are not promoted to valid scores. The original results document
is retained as `evidence/cs781_results.docx` for comparison. The report typesets the data; it does not use
screenshots as figures.

`local_variant_tables.json` and `learning_curves.json` preserve executed notebook observations.
`evidence_sha256.json` hashes their original sources. The extraction uses original notebook snapshots retained before presentation cleanup. These output-bearing originals are not distributed:
the code package contains clean copies. Exact winning-run settings were not retained for every phase;
the per-phase configuration provenance describes the missing information. Source defaults and later
saved cells are not represented as exact historical winning configurations.

To regenerate original vector figures and tables from the supplied evidence:

```powershell
python -m pip install matplotlib
python report/support/build_tables.py
python report/support/make_figures.py
```

The figures are a system diagram, document-length profiles, retained learning trajectories and the
padding diagnostic. No missing training curve is reconstructed. `collect_evidence.py` documents the
initial extraction process; it needs the original working-repository notebook outputs and is not needed
to rebuild the supplied PDF.

To repeat the random-model padding check, install the code dependencies and run:

```powershell
python report/support/diagnose_padding.py
```

This check uses tiny untrained A/B models on CPU. It demonstrates an implementation property, without
estimating the fraction of B's competition error caused by padding.

To audit the PDF, install `pypdf`, `pymupdf` and `pillow`, then run `python report/support/audit_report.py`
after compilation. This creates rendered review images locally. The source assembly and packaging
scripts document preparation in the original workspace; they are not required for model training or PDF
compilation from the submitted files.

## Code repository

The repository URL supplied by the student is
[maheshwari-nitin/cs781](https://github.com/maheshwari-nitin/cs781).
The LLM declaration records code, tuning, analysis and report assistance; it should be reviewed by the
student as part of taking responsibility for the submission.
