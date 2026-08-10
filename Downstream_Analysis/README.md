# Downstream analysis

This directory contains notebooks and scripts for analysing olfactory-receptor probability matrices.

## Prediction dataset

Prediction outputs are not stored in Git; they are published in the project's Zenodo record, alongside the ESM-C protein embeddings. Record ID and per-asset SHA-256 checksums live in `zenodo_manifest.json` at the repository root.

The dataset carries **probabilities only**, for `ASR_Predictions` and `Reference_Tree_Predictions`:

```
runs/<Category>/<run files>        per-run probabilities, all five runs
aggregated/<Category>/<matrix>     median-probability wide matrices
model_run_evaluation_metrics.csv   per-run evaluation metrics
MANIFEST.json                      per-file sizes and SHA-256 checksums
```

The pipeline is: **five runs → median per (protein_id, smiles_id) → threshold applied in the notebook.** The median matrices are the analysis input. The per-run files are published so the medians can be independently reproduced rather than taken on trust; regenerating from them yields byte-identical matrices.

No binary call table is distributed. Each notebook derives its own calls from the median matrix at whichever threshold it is testing, so the threshold stays visible at the point of use instead of being frozen into a shipped file.

### Download

```bash
python scripts/download_zenodo_data.py                 # predictions + pooled embeddings
python scripts/download_zenodo_data.py --list          # show every asset
```

This verifies each archive's checksum and extracts it into `Downstream_Analysis/predictions/`, which is ignored by Git.

### Regenerate the medians from the runs

```bash
python Downstream_Analysis/scripts/aggregate_prediction_runs.py
```

It reads `predictions/runs/<Category>/`, falling back to a local `predictions/archive/Predictions_M2OR-*` export when present, and writes the median-probability matrices to `predictions/aggregated/<Category>/`. Use `--category` to process a single category and `--dry-run` to preview.

## Create the Zenodo release archive

Maintainers can package a new version with:

```bash
python Downstream_Analysis/scripts/prepare_zenodo_predictions.py --version v1
```

It reads the per-run files from the latest `predictions/archive/Predictions_M2OR-*` export, drops each run's `prediction` column (a 0.5-thresholded derivative that would conflict with the thresholds used in the notebooks), copies probability values as text so published numbers are byte-identical to the model output, and writes a ZIP plus a `.sha256` file to the ignored `release/` directory at the repository root. Pass `--no-medians` to publish the runs alone.

The protein embeddings for the same record are built by `Model/Embeddings/prepare_zenodo_embeddings.py`, which writes to the same `release/` directory.

Upload every `.zip` and `.sha256` from `release/` to Zenodo, publish the record, then fill in `zenodo_record_id` and each asset's `sha256` in `zenodo_manifest.json`. Note that rebuilding an archive changes its checksum (ZIPs embed timestamps), so re-sync the manifest if you regenerate anything after uploading.
