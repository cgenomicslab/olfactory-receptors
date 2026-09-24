# Downstream analysis

What the predicted binding probabilities say. This is the bulk of the paper, and none of it
needs a GPU.

## Notebooks

**Living receptors** — [`notebooks/reference_tree/`](notebooks/reference_tree/)

| Notebook | Shows |
|---|---|
| [`figure2_activation_code.ipynb`](notebooks/reference_tree/figure2_activation_code.ipynb) | **Figure 2**, Table 1, Extended Data 2–4, and the pathway numbers of Supplementary Table 10 |
| [`supplementary/`](notebooks/reference_tree/supplementary/) | S1–S6: model errors, the p ≥ 0.50 matrix, other views of the chemistry, pathway figures, barcodes and perception, aquatic origin |

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
| `activation_code.py` | the human activation matrix, its clusters and the odorant chemistry |
| `activation_figures.py` | the figures of the reference-tree notebooks |
| `ancestral_sets.py` | loads ancestral repertoires, walks the tree, builds the node sets |
| `plotting_functions.py` | the shared plot style |
| `aggregate_prediction_runs.py` | five runs → per-pair median |
| `prepare_zenodo_predictions.py` | packages the matrices for release |
