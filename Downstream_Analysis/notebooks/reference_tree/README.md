# The 433 living human receptors

Start with [`figure2_activation_code.ipynb`](figure2_activation_code.ipynb). It builds
Figure 2, Table 1 and Extended Data Figures 2–4, and shows every panel inline.

The other three notebooks are supporting analyses that the paper draws on but does not
give a figure of its own:

- [`model_errors.ipynb`](model_errors.ipynb) — false positives and false negatives, by
  receptor class, by chemistry, and against phylogenetic distance.
- [`barcode_perception.ipynb`](barcode_perception.ipynb) — molecules that switch on
  exactly the same receptors: do they smell the same? With a chemical-similarity control,
  because the obvious objection is that they are simply similar molecules.
- [`biosynthetic_pathways.ipynb`](biosynthetic_pathways.ipynb) — whether the activation
  clusters line up with biosynthetic origin, against a clusters-from-chemistry-alone control.

## Output folders

| Folder | Holds |
|---|---|
| `Figures/` | the panels, as SVG |
| `Tables_barcodes/`, `metadata/` | the numbers behind them |

Run `figure2_activation_code.ipynb` before `model_errors.ipynb`: the first writes
`metadata/exp_pred_common_pairs.csv`, which the second reads.
