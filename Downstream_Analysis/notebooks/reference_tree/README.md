# The 433 living human receptors

[`figure2_activation_code.ipynb`](figure2_activation_code.ipynb) builds everything the paper
shows from these receptors, in the order the paper uses it. Everything else is in
[`supplementary/`](supplementary/).

Set up and activate the environment as in the [main README](../../../README.md), and fetch the
prediction matrices with `python scripts/download_zenodo_data.py`. Run the main notebook first:
`supplementary/S1` reads its `Tables/exp_pred_common_pairs.csv`.

Most of the code lives in two modules in `Downstream_Analysis/scripts/`: `activation_code.py`
(loading, statistics, clustering, chemistry) and `activation_figures.py` (the figures). The
binding calls come from `calibration.py` in the same folder.

## Main notebook

| Section | Paper item | Files |
|---|---|---|
| 1 · Model check | Extended Data Fig. 2a–d, Supplementary Table 4 | `Figures/threshold_calibration.svg`, `confusion_matrix_common_pairs.svg`, `probability_density_by_confusion_matrix_category.svg`, `proportions_scatter.svg`; `Tables/exp_pred_common_pairs.csv` |
| 2 · The hybrid matrix and its six clusters | Fig. 2a, Extended Data Fig. 3 | `Figures/heatmap_hybrid_density.svg`, `heatmap_hybrid_density_tree_order.svg` |
| 3 · Ligands per receptor by class | Fig. 2b, Supplementary Table 2 | `Figures/class_tuning_breadth_hist.svg`; `Tables/class_tuning_breadth.csv` |
| 4 · What each class reads | Fig. 2c and inset, Fig. 2d | `Figures/chemspace_hybrid_density_groups.svg`, `chemspace_hybrid_density_effects_cliffs.svg`, `odour_tags_hybrid_density_exclusive.svg`; `Tables/chemspace_hybrid_density_descriptors.csv`, `odour_tags_hybrid_density_percentages.csv` |
| 5 · What the clusters smell of | Fig. 2e, Table 1, Supplementary Table 5 | `Figures/enrichbar_hybrid_density.svg`, `enrichment_table_hybrid_density.svg`; `Tables/enrichment_hybrid_density.csv`, `enrichment_table_hybrid_density.csv` |
| 6 · Cluster chemistry | Extended Data Fig. 4a–c, Table 1, Supplementary Tables 5 and 6 | `Figures/pcoa_hybrid_density.svg`, `chemspace_clusters_hybrid_density_groups.svg`, `chemspace_clusters_hybrid_density_effects.svg`; `Tables/chemspace_clusters_hybrid_density_descriptors.csv`, `chemspace_clusters_hybrid_density_summary.csv` |
| 7 · Biosynthetic pathway | Results text, Supplementary Table 10 | `Tables/05_pathway_recovery.csv` |

`scripts/build_extended_figures.py` and `scripts/build_supplementary_tables.py` at the
repository root read these files by name, so rename them in both places or not at all.

## Supplementary notebooks

| Notebook | What it covers |
|---|---|
| [`S1.model_errors`](supplementary/S1.model_errors.ipynb) | probability distributions, measured against predicted counts, the M2OR mutant flag against SwissProt, and the false positives and false negatives by class, chemistry and tree distance |
| [`S2.baseline_matrix`](supplementary/S2.baseline_matrix.ipynb) | the same clusters on predictions alone at p ≥ 0.50, for comparison |
| [`S3.chemistry_other_views`](supplementary/S3.chemistry_other_views.ipynb) | tuning breadth as a strip and a density, class descriptor violins, fold changes between the class sets, odour descriptors of the full class sets, cluster descriptor violins |
| [`S4.pathway_figures`](supplementary/S4.pathway_figures.ipynb) | the figures for section 7 |
| [`S5.barcode_perception`](supplementary/S5.barcode_perception.ipynb) | whether odorants with identical receptor barcodes share odour descriptors, with a structural-similarity control |
| [`S6.aquatic_origin`](supplementary/S6.aquatic_origin.ipynb) | whether the odorants each class reads come from aquatic organisms; needs `Chemicals/results/aquatic_sources_panel.csv` |

They write to `supplementary/Figures/` and `supplementary/Tables/`.

## Which numbers to report

- Binding calls come from the density fill: an isotonic map from score to measured binding
  frequency sets how many cells are called, rank sets which, and the measured pairs are
  written over the top. The cut it prints is an output of the fill.
- The rate-matched cut (0.915) is used only for the confusion matrix and the per-pair metrics
  in section 1.
- The main matrix is `hybrid_density`. `baseline` (S2) is for comparison only.
