# The 433 extant human receptors

The predictions cover the 584 receptors of the six-species reference tree; these notebooks
use its 433 human rows. They write tables to `results/` and figures to `figures/`.

Fetch the predictions first (`python scripts/download_zenodo_data.py`), then run the
notebooks in order: `e03`–`e05` read the matrix and clusters written by `e02`.

| Notebook | Computes | Writes `results/` | Used in |
|---|---|---|---|
| [`e01_model_check`](e01_model_check.ipynb) | the measured pairs against their predictions: the rate-matched cut and the classification metrics | `model_check_*.csv` | Extended Data Fig. 2 |
| [`e02_activation_matrix`](e02_activation_matrix.ipynb) | the hybrid matrix (density fill plus the measurements), the six clusters, their odour-descriptor enrichment | `hybrid_matrix.csv`, `density_fill.csv`, `odorant_clusters.csv`, `cluster_tag_enrichment.csv` | Fig. 2a, 2e, Extended Data Fig. 3 |
| [`e03_receptor_classes`](e03_receptor_classes.ipynb) | ligands per receptor by class; functional groups, descriptors and odour descriptors of the odorants each class reads | `class_*.csv` | Fig. 2b–d |
| [`e04_cluster_chemistry`](e04_cluster_chemistry.ipynb) | functional groups and descriptors of each cluster, the principal coordinates, Table 1 | `cluster_*.csv`, `table1_clusters.csv` | Table 1, Extended Data Fig. 4 |
| [`e05_biosynthetic_pathway`](e05_biosynthetic_pathway.ipynb) | how well the clusters, and clusters built from descriptors alone, recover biosynthetic pathway | `pathway_*.csv` | Results text |

The code the notebooks call is in `Downstream_Analysis/scripts/`: `activation_code.py`
(matrix, statistics, clustering, chemistry), `calibration.py` (the density fill) and
`activation_figures.py` (the drawing functions). Functional groups are defined once, in
`scripts/functional_groups.py`.

Binding calls come from the density fill: an isotonic map from score to measured binding
frequency sets how many cells are called, the score sets which, and measured pairs are
written over the fill. The rate-matched cut of `e01` (0.915) is used only for the model check.
