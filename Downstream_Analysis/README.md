# Downstream analysis

What the predicted binding probabilities say. This is the bulk of the paper, and none of it
needs a GPU.

## Notebooks

**Living receptors** — [`notebooks/reference_tree/`](notebooks/reference_tree/)

| Notebook | Shows |
|---|---|
| [`figure2_activation_code.ipynb`](notebooks/reference_tree/figure2_activation_code.ipynb) | **Figure 2**, Table 1, Extended Data 2–4 |
| [`model_errors.ipynb`](notebooks/reference_tree/model_errors.ipynb) | where the model gets it wrong, and whether the errors have structure |
| [`barcode_perception.ipynb`](notebooks/reference_tree/barcode_perception.ipynb) | do identical receptor barcodes mean identical smell? |
| [`biosynthetic_pathways.ipynb`](notebooks/reference_tree/biosynthetic_pathways.ipynb) | pathway against activation cluster |

**Reconstructed ancestors** — [`notebooks/ancestral/`](notebooks/ancestral/)

| Notebook | Shows |
|---|---|
| [`figure3_ancestral_origin.ipynb`](notebooks/ancestral/figure3_ancestral_origin.ipynb) | **Figure 3**, Extended Data 5–6 |

Run them top to bottom. Figures appear inline; the versions used in the manuscript are
written to each folder's `Figures/`, and the numbers behind them to `Tables/`.

## Getting the data first

The probability matrices are too large for Git.

```bash
python scripts/download_zenodo_data.py
```

That gives `predictions/aggregated/` (the per-pair medians the notebooks read) and
`predictions/runs/` (the five individual runs).

To rebuild the medians from the runs yourself:

```bash
python Downstream_Analysis/scripts/aggregate_prediction_runs.py
```

## Shared code

`scripts/` holds what the notebooks import.

| File | Does |
|---|---|
| `calibration.py` | turns probabilities into binding calls by density fill |
| `ancestral_sets.py` | loads ancestral repertoires, walks the tree, builds the node sets |
| `plotting_functions.py` | the shared plot style |
| `aggregate_prediction_runs.py` | five runs → per-pair median |
| `prepare_zenodo_predictions.py` | packages the matrices for release |
