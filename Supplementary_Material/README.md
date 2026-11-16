# Supplementary Material

The analyses behind the Supplementary Notes and Figures, and the Supplementary Tables.

Set up the environment as in the [main README](../README.md), fetch the predictions
(`python scripts/download_zenodo_data.py`) and run the notebooks of
[`Downstream_Analysis/`](../Downstream_Analysis/README.md) first: the notebooks here read the
tables they write. Each folder has one notebook, which writes its tables to `Tables/` and its
figures to `Figures/` in that folder.

| Folder | Asks | Used in |
|---|---|---|
| [`model_comparison/`](model_comparison/model_comparison.ipynb) | how the binding model compares with baselines (odorant and receptor binding rates; gradient boosting on the same frozen embeddings) on random, unseen-receptor and unseen-odorant splits | Supplementary Note 1, Supplementary Fig. 1 |
| [`cluster_number/`](cluster_number/cluster_number.ipynb) | why six activation clusters: k from 2 to 12, re-clustering of random subsamples, odorants between clusters, other clustering methods, and a test without any clustering | Supplementary Note 2, Supplementary Fig. 2 |
| [`calling_rule/`](calling_rule/calling_rule.ipynb) | whether the extant and ancestral results hold when the calls are made with half to twice the calibrated number, with fixed probability cuts, or without the measurements | Supplementary Note 3, Supplementary Figs. 3 and 4 |
| [`model_runs/`](model_runs/model_runs.ipynb) | each training run on its own held-out test pairs, and the results from each of the five runs alone | Supplementary Notes 1 and 3 |
| [`ancestral_confidence/`](ancestral_confidence/ancestral_confidence.ipynb) | the posterior probability of every reconstructed sequence and the bootstrap support of every node | Supplementary Note 4, Supplementary Fig. 5 |

`model_runs` reads the per-run predictions of the Zenodo archive, `model_comparison` the
M2OR embeddings (`python scripts/download_zenodo_data.py --embeddings-full`) and
`ancestral_confidence` the per-node runs (`--asr`).

`robustness_checks.py` holds what the notebooks share: the other calling rules, and the code
that recomputes the extant and ancestral results on any set of calls, so every check
recomputes a result the same way.

## Supplementary Tables

```bash
python Supplementary_Material/supplementary_tables/build_supplementary_tables.py \
    --out Supplementary_Material/supplementary_tables/Supplementary_Tables.xlsx
```

Writes one tab per Supplementary Table from the tables of `Downstream_Analysis/`,
`Chemicals/results/` and the folders above. Each tab opens with the table's title.
`--legends <file>` replaces the titles with full legends, one paragraph per table starting
with `Supplementary Table N |`.
