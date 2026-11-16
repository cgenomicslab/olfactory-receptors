# The 432 reconstructed ancestors

Fetch the predictions first (`python scripts/download_zenodo_data.py`), then run the
notebooks in order. They write tables to `results/` and figures to `figures/`.

| Notebook | Computes | Writes `results/` | Used in |
|---|---|---|---|
| [`a01_node_repertoires`](a01_node_repertoires.ipynb) | repertoires, turnover and chemistry of nodes 0–2 | `repertoire_turnover.csv` | Extended Data Fig. 5b |
| [`a02_pattern_contrasts`](a02_pattern_contrasts.ipynb) | presence pattern of every ligand over nodes 0–2; class I side against class II side | `group_sizes.csv`, `ligand_patterns.csv` | Fig. 3b, 3c |
| [`a03_functional_decay`](a03_functional_decay.ipynb) | ligand count against distance from the root, every node | `ligands_vs_root_distance.csv`, `root_distance_statistics.csv` | Extended Data Fig. 5a, 5c |
| [`a04_parent_child_decay`](a04_parent_child_decay.ipynb) | parent–child Jaccard against branch length, exponential fit, threshold sweep | `edge_table.csv`, `decay_fit.csv` | Fig. 3f, Extended Data Fig. 5d |
| [`a05_ten_node_lineage`](a05_ten_node_lineage.ipynb) | ten nodes; the two founding duplications: gain-set chemistry, shift vectors, null, PCA, cross-checks | `topology.csv`, `lineage_functional_groups.csv`, `dup_chemistry.csv`, `shift_plane.csv`, `orthogonality*.csv`, `pca_*.csv`, `loss_selectivity.csv`, `table1_nodes.csv`, `table1_stages.csv` | Fig. 3d, 3e, Extended Data Fig. 6, Supplementary Tables 7–8 |
| [`a06_code_trajectory`](a06_code_trajectory.ipynb) | the code on one time axis, without the measurements, ultrametric and raw; nulls and sensitivity | `trajectory_raw_axis.csv` | Fig. 3i (raw axis) |
| [`a07_combinatorial_code_origin`](a07_combinatorial_code_origin.ipynb) | the first three stages; the trajectory with the measurements written over the calls | `trajectory.csv` | Fig. 3g–i |

The notebooks write more tables than the ones listed: tables that nothing else reads are
written when a notebook runs and are not kept in the repository.

The code the notebooks call is in `Downstream_Analysis/scripts/`: `ancestral_sets.py`
(repertoires, the tree walk, the node sets), `calibration.py` (the density fill) and
`ancestral_figures.py` (the drawing functions for the panels of Fig. 3 and Extended Data
Figs. 5–6). Functional groups are defined once, in `scripts/functional_groups.py`.

Node numbers: the tree file names the common ancestor of the human receptors `node_1`; the
paper numbers it 0, so tree `node_k` is paper node k − 1. Figures use the paper's numbers and
tables carry both (`node_fig`), except `lineage_functional_groups.csv`, whose columns are tree ids.
