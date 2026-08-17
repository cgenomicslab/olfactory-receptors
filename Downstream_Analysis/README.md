# Downstream analysis

Notebooks that read the predicted olfactory-receptor × odorant probability matrices and
ask what they say — about extant human receptors, and about the reconstructed ancestors.

```
notebooks/reference_tree/   extant human receptors
notebooks/ancestral/        reconstructed ancestral nodes
scripts/                    shared code the notebooks import
predictions/                the probability matrices (from Zenodo, not Git)
reports/                    written-up results
```

Each notebooks folder has its own README with the detail. Start there.

## The notebooks

**`notebooks/reference_tree/`** — the 433 human ORs.

| | |
|---|---|
| `01.model_behaviour` | Derives the binding threshold from the measured pairs. Read this one first. |
| `02.OR_human_landscape` | The receptor × odorant landscape: activation clusters, chemistry, class I vs class II. |
| `03.fp_fn_exploration` | Where the model gets it wrong, and whether the errors are structured. |
| `04.barcode_perception` | Receptor activation barcodes against perceived odour quality. |

**`notebooks/ancestral/`** — the reconstructed nodes, `01` through `05`. Node repertoires,
activation-pattern contrasts, decay with tree depth, the parent–child functional half-life,
and a ten-node walk down one lineage.

## One thing to know before reading any number

**The two folders use different thresholds, on purpose.**

The pipeline is: five model runs → median probability per pair → threshold applied in the
notebook. No binary call table is shipped, so every notebook makes its own call visible at
the point of use.

`reference_tree/` cuts at **0.915**. That is calibrated: it is where the predicted binding
proportion matches the measured one across the 23,782 experimentally tested human pairs.

`ancestral/` cuts at **0.5**. Reconstructed sequences score systematically lower than real
ones, and at 0.915 the three core ancestral nodes are left with 5, 9 and 1 ligands. At 0.5
they hold 177, 130 and 113. Comparisons there are internal — node against node — so the
same cut applies to both sides of every contrast, and repertoire sizes should be read as
relative rather than as absolute binding counts.

`ancestral/04` tests whether its result survives the choice: sweeping the cut from 0.3 to
0.8 moves the correlation by 0.03, and a threshold-free weighted Jaccard shows the same
decay.

## Getting the data

Prediction outputs are not in Git. They live in the project's Zenodo record alongside the
ESM-C protein embeddings; the record ID and per-asset SHA-256 checksums are in
`zenodo_manifest.json` at the repository root.

```bash
python scripts/download_zenodo_data.py          # predictions + pooled embeddings
python scripts/download_zenodo_data.py --list   # show every asset
```

This verifies each archive's checksum and extracts into `predictions/`, which is
gitignored. The dataset carries probabilities only, for `ASR_Predictions` and
`Reference_Tree_Predictions`:

```
runs/<Category>/<run files>        per-run probabilities, all five runs
aggregated/<Category>/<matrix>     median-probability wide matrices
model_run_evaluation_metrics.csv   per-run evaluation metrics
MANIFEST.json                      per-file sizes and SHA-256 checksums
```

The per-run files are published so the medians can be reproduced rather than taken on
trust — regenerating from them yields byte-identical matrices:

```bash
python Downstream_Analysis/scripts/aggregate_prediction_runs.py
```

Use `--category` for a single category and `--dry-run` to preview.

## Shared code

`scripts/` holds what the notebooks import, so the analysis stays in the notebooks and the
plumbing does not get copy-pasted between them.

- `ancestral_sets.py` — loads the matrices, builds the ancestral node sets and activation
  patterns, and provides the tree walks (`decay_table`, `edge_table`, `node_topology`) plus
  the RDKit and odour-tag helpers. Also `save_fig` / `save_table`, which is how the
  ancestral notebooks write to `Figures/` and `Tables/`.
- `plotting_functions.py` — the rcParams every notebook uses, so the figures match.
- `aggregate_prediction_runs.py` — five runs to median matrices.
- `prepare_zenodo_predictions.py` — packages a release.

## Packaging a Zenodo release

```bash
python Downstream_Analysis/scripts/prepare_zenodo_predictions.py --version v1
```

Reads the per-run files from the latest `predictions/archive/Predictions_M2OR-*` export,
drops each run's `prediction` column (a 0.5-thresholded derivative that would conflict with
the thresholds used in the notebooks), copies probability values as text so published
numbers are byte-identical to the model output, and writes a ZIP plus a `.sha256` to the
gitignored `release/` directory at the repository root.  `--no-medians` publishes the runs
alone. Protein embeddings for the same record come from
`Model/Embeddings/prepare_zenodo_embeddings.py`, which writes to the same place.

Upload every `.zip` and `.sha256` from `release/`, publish, then fill in
`zenodo_record_id` and each asset's `sha256` in `zenodo_manifest.json`. Rebuilding an
archive changes its checksum (ZIPs embed timestamps), so re-sync the manifest if you
regenerate anything after uploading.
