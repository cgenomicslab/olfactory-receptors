# Extant human receptors

The 433 human olfactory receptors of the reference tree against 754 odorants: how the model
behaves where we can check it, what the predicted landscape looks like, where the errors
sit, and whether receptor activation patterns carry perceptual meaning.

## How to run

```bash
conda activate olfactory-receptors
python scripts/download_zenodo_data.py       # from the repository root, once
python scripts/check_reproducibility.py
jupyter lab                                  # then run 01 -> 05 in order
```

**Run them in order.** `01` derives the threshold everything else reports and writes
`metadata/exp_pred_common_pairs.csv`, which `03` reads. This is a real dependency: if you
change the threshold in `01` and do not re-run it, `03` will silently disagree with it.

| Notebook | Question | Runtime |
|---|---|---|
| `01.model_behaviour` | Where does the binding threshold come from, and how good is the model at it? | ~3 min |
| `02.OR_human_landscape` | The receptor × odorant landscape: activation clusters, odour enrichment, class I vs class II chemistry. | ~15 min (PAM) |
| `03.fp_fn_exploration` | Where the model is wrong, and whether the errors are structured or random. | ~3 min |
| `04.barcode_perception` | Do molecules that activate the same receptors smell the same? | ~10 min |
| `05.biosynthetic_pathways` | Do the activation clusters recover where the molecules come from biosynthetically? | ~10 min |

Each notebook's **Paths** cell is the only block to edit if your layout differs; everything
downstream reads those variables.

## The two numbers, and which does what

| | value | what it is for |
|---|---|---|
| **rate-matched cut** | `0.9150` | **reporting only** — the per-pair classification operating point |
| **density cut** | `0.8947` (an output) | **construction** — what actually fills the matrix |

`01` derives the rate-matched cut. **23,782 pairs** can be checked: 409 of the 433
receptors against 652 odorants have a measured M2OR result, joined by exact amino-acid
sequence match. Of those, **4.7431%** bind. The calibrated threshold is the cut where the
model predicts that same proportion:

```
measured binding proportion       4.7431%
predicted proportion at p > 0.5  17.1895%     <- the default over-calls by 3.6x
calibrated threshold             0.9150       <- predicted proportion = 4.7431%
AUROC (threshold-free)           0.9699
```

Two things make 0.9150 more than a convenience. MCC and F1 both peak independently at
0.920, so the rate-matching cut lands where the standard metrics want it anyway. And at an
exact rate match false positives must equal false negatives, which is what the confusion
matrix shows:

```
TP    672   (2.8%)        FP   456   (1.9%)
FN    456   (1.9%)        TN 22,198  (93.3%)
```

**But it is not what builds the matrices.** Recall at 0.9150 is ~0.6, so thresholding the
full grid drops roughly 40% of real binders. `02`, `04` and `05` all build their activation
matrix with the density fill of [`../../scripts/calibration.py`](../../scripts/calibration.py)
instead, where the cut is an output. See
[`../../README.md`](../../README.md#binarisation-how-a-probability-becomes-a-binding-call).

The ancestral notebooks use the same fill but no experimental overlay — see
[`../ancestral/README.md`](../ancestral/README.md).

## The hybrid matrix

`02`, `04` and `05` all run on the same object, built the same way:

1. apply the isotonic map to every **untested** cell → its calibrated binding probability
2. `N` = sum of those probabilities = expected number of true binders
3. rank the untested cells by model score, set the top `N` to 1
4. overlay the 23,782 measurements — a measured value always beats a predicted one

The experimental override is keyed on **exact amino-acid sequence identity**, never on
UniProt accession. Mutant receptors in M2OR share the parent accession but differ in
sequence, so sequence matching excludes them automatically; accession matching would fold
mutant measurements onto wild-type predictions and inflate the correction.

`02` also builds a `baseline` matrix at p ≥ 0.50 for reference, written to `Figures/` only.

**Orphan receptors.** The hybrid matrix leaves more orphans than a plain threshold at the
same cut would. That is the overlay working: a receptor whose only above-cut cell was tested
and measured negative loses it. Those are model false positives M2OR already disproved.

## Clustering

`02` and `05` cluster molecules by **Jaccard distance on the activation barcode + PAM,
k = 6**, with hypergeometric odour-descriptor enrichment, BH corrected. Inside each cluster,
columns are ordered by average-linkage hierarchical clustering on the same distance — that
is **seriation only**: it decides where a column is drawn, never which cluster it belongs
to. No sub-clusters are defined and no dendrogram is shown, because PAM never computed one.

## What the notebooks find

`02` section 9 partitions odorants three ways by which receptor class binds them: **115
class I only, 218 shared, 113 class II only**. A molecule enters a class's bag when it
activates at least `r = 1/62` of that class — one Class I receptor, or six Class II
receptors. Equal rates rather than equal counts, so the 62/371 class-size difference does
not decide membership.

`03` finds the errors are not random. Class I receptors carry a higher false-positive rate
than class II (2.77% vs 1.90%, χ² p = 2.8e-10), and false negatives cluster: 177 receptors
carry at least one, and 61 of those also have true positives, so the model is not simply
blind to them.

`04` asks whether identical receptor barcodes imply identical perception. 272 molecules
fall into 65 groups sharing a barcode; 389 are unique and 93 activate nothing. Silent
molecules are excluded from every collision analysis — an all-zero barcode is an *absence*
of a code, not a shared code. Molecules with the same barcode do share more odour
descriptors than random pairs, but the effect is modest and survives controlling for
structural similarity (p = 0.002 among structurally dissimilar pairs). **Read it as a weak
signal, not a demonstration.**

`05` joins the ligands to COCONUT (2026-08 release) on InChIKey for NPClassifier pathway
labels. 536 of the 754 are in COCONUT, 481 carry a pathway, and 427 fall into the five
classes analysed after dropping Polyketides (12) and Carbohydrates (3) — with all seven, 20
of 42 expected cells sit below 5 and chi-square is invalid. The mix is dominated by fatty
acids (205).

Both clusterings beat the permutation floor by a wide margin, but the ordering is what
matters:

```
descriptor clusters   Cramer V = 0.569     floor 0.107 (95th 0.137)
receptor clusters     Cramer V = 0.356     floor 0.107
```

Descriptors alone recover pathway better than the receptor code does — and since those
clusters are built from the same descriptors that drive pathway assignment, 0.569 is a
**ceiling rather than a rival method**. NPClassifier infers pathway from structure and the
activation profiles are predicted from structure, so part of the association is two
structure-derived labels agreeing. The claim is that the clusters are chemically coherent
enough to recover biosynthetic origin — not that the receptor code carries biosynthetic
information beyond chemistry.

## Outputs

```
Figures/           01-03, 05: activation heatmaps, PCoA, enrichment bars, chemistry, confusion matrix
Figures_barcodes/  04
Tables_barcodes/   04: barcode groups, shared descriptors, structure control
metadata/          exp_pred_common_pairs.csv - written by 01, read by 03
```

`metadata/exp_pred_common_pairs.csv` holds the 23,782 measured pairs with their predicted
probability and confusion-matrix category, called at 0.9150:

| column | meaning |
|---|---|
| `uniprot_id`, `seq_id`, `sequence` | the receptor |
| `smiles_id`, `SMILES` | the odorant |
| `experimental` | measured label, 0/1, maxed over replicates |
| `predicted` | binary call at 0.9150 |
| `probability` | raw median model probability |
| `mutation`, `OR_class` | receptor metadata |

## Inputs

In the repository:

| path | what it gives |
|---|---|
| `Model/Data_Preparation/processed/m2or_pairs_model.csv` | measured pairs |
| `Phylogenetic_Analysis/data/ReferenceTree/PF13853.*.fa` | sequences, for the join |
| `Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv` | Class I / II assignment |
| `Phylogenetic_Analysis/data/HumanTree/human433_MFP_reorder.tree` | phylogenetic row order in `02` |
| `Chemicals/odor_datasets/inchi_odors_smiles.csv` | odour descriptors |
| `Chemicals/odor_datasets/ligand_pathways.csv` | NPClassifier pathways for `05` |

From the Zenodo record — see [`../../README.md`](../../README.md#getting-the-data):

- `predictions/aggregated/Reference_Tree_Predictions/reference_median_probability_wide.csv`

### Patristic distances in `03`

The final section of `03` compares false-positive rate against phylogenetic distance. It
computes the 433 × 433 patristic distance matrix from
`Phylogenetic_Analysis/data/HumanTree/human433_MFP_reorder.tree` directly, in one tree
walk (~0.01 s), rather than reading a precomputed array.

This used to load a `.npy` from outside the repository and that section did not run on a
fresh clone. The computed matrix is numerically identical to the one it replaced
(max |difference| 3.6e-15 — the old array was built from this same tree), so the published
numbers are unchanged. The whole notebook now runs from a clean clone.
