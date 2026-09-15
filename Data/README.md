# Data

`m2or_pairs.csv` — the M2OR export (<https://m2or.chemsensim.fr/>). Everything in this
repository is built from it.

[`data.ipynb`](data.ipynb) explores it. Figures in `figures/`.

| | |
|---|---|
| assay observations | 53,444 |
| positives | 3,124 (5.9%) |
| receptor sequences | 1,402 (589 wild type, 813 mutant) |
| odorants | 771 |

`Sequence` is the join key, not `UniProt ID` — a wild type shares its accession with its
mutants.

Dropping mixtures leaves 52,175 rows over 1,399 sequences and 754 odorants. That happens in
[`Model_Inputs/Data_Preparation/`](../Model_Inputs/Data_Preparation/).
