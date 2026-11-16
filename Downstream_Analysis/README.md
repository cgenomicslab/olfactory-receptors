# Downstream analysis

What the predicted binding probabilities say about extant and reconstructed receptors.

Set up the environment as in the [main README](../README.md), fetch the prediction matrices,
then run the notebooks of each folder in order:

```bash
python scripts/download_zenodo_data.py     # the prediction matrices, 33 MiB
```

| Folder | Receptors | Notebooks | Used in |
|---|---|---|---|
| [`extant_receptors/`](extant_receptors/) | the 433 extant human receptors (the human rows of the 584-receptor reference-tree matrix) | `e01`–`e05` | Fig. 2, Table 1, Extended Data Figs. 2–4 |
| [`ancestral_receptors/`](ancestral_receptors/) | the 432 reconstructed ancestors of the human tree | `a01`–`a07` | Fig. 3, Extended Data Figs. 5–6 |

Each notebook writes its tables to `results/` and its figures to `figures/` in its folder.

## The predictions

`predictions/aggregated/` holds the per-pair median of the five model runs, the matrix every
notebook reads; `predictions/runs/` holds the five runs. Both come from Zenodo. To rebuild the
medians from the runs:

```bash
python Downstream_Analysis/scripts/aggregate_prediction_runs.py
```

The predictions were made with the binding model
[PLinter](https://github.com/cgenomicslab/PLinter) (config `configs/esmc.yaml`), trained five
times on `Model_Inputs/Data_Preparation/processed/m2or_pairs_model.csv`, each run on its own
stratified 80/10/10 split. Each trained model scored the 754 odorants against the 584
reference-tree receptors (`Phylogenetic_Analysis/data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa`)
and the 432 ancestors (`Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/hagfish_human_ASR.final.fasta`).

## Shared code

| File in `scripts/` | Does |
|---|---|
| `calibration.py` | turns probabilities into binding calls by density fill |
| `activation_code.py` | the extant-receptor matrix, its clusters, the statistics behind Fig. 2 |
| `activation_figures.py` | the drawing functions for Fig. 2 and Extended Data Figs. 2–4 |
| `ancestral_sets.py` | ancestral repertoires, the tree walk, the node sets |
| `ancestral_figures.py` | the drawing functions for Fig. 3 and Extended Data Figs. 5–6 |
| `aggregate_prediction_runs.py` | five runs → per-pair median |
| `prepare_zenodo_predictions.py` | packages the prediction matrices for Zenodo |

Functional groups are defined once, for the whole repository, in `scripts/functional_groups.py`.
