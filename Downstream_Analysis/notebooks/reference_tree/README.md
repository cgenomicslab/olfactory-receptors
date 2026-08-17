# Extant human receptors

The 433 human olfactory receptors of the reference tree against 754 odorants: how the
model behaves where we can check it, what the predicted landscape looks like, where the
errors sit, and whether receptor activation patterns carry perceptual meaning.

Run them in order. `01` derives the threshold everything else uses and writes the table
`03` reads, so it is not optional.

| Notebook | Question |
|---|---|
| `01.model_behaviour` | Where does the binding threshold come from, and how good is the model at it? |
| `02.OR_human_landscape` | The receptor × odorant landscape: activation clusters, odour enrichment, class I vs class II chemistry. |
| `03.fp_fn_exploration` | Where the model is wrong, and whether the errors are structured or random. |
| `04.barcode_perception` | Do molecules that activate the same receptors smell the same? |

## The threshold

The model outputs probabilities. Turning them into binding calls needs a cut, and `01`
derives it rather than assuming one.

**23,782 pairs** can be checked: 409 of the 433 receptors against 652 odorants have a
measured M2OR result, joined by exact amino-acid sequence match. Of those, **4.7431%**
bind. The calibrated threshold is the cut where the model predicts that same proportion:

```
measured binding proportion       4.7431%
predicted proportion at p > 0.5  17.1895%     ← the default over-calls by 3.6x
calibrated threshold             0.9150       ← predicted proportion = 4.7431%
AUROC (threshold-free)           0.9699
```

Two things make 0.9150 more than a convenience. MCC and F1 both peak independently at
0.920, so the rate-matching cut lands where the standard metrics want it anyway. And at
an exact rate match false positives must equal false negatives, which is what the
confusion matrix shows:

```
TP    672   (2.8%)        FP   456   (1.9%)
FN    456   (1.9%)        TN 22,198  (93.3%)
```

This is not the same cut the ancestral notebooks use. See `../ancestral/README.md` — it is
calibrated on extant sequences and does not transfer to reconstructions.

## What the notebooks find

`02` builds four activation matrices — predictions at 0.5 and at 0.9150, each with and
without measured M2OR values substituted in — and clusters each one (Jaccard + PAM, k=6)
with hypergeometric odour-descriptor enrichment, BH corrected. The hybrid calibrated
matrix is the one the figures use. Section 9 partitions odorants three ways by which
receptor class binds them: 115 class I only, 218 shared, 113 class II only.

`03` finds the errors are not random. Class I receptors carry a higher false-positive
rate than class II (2.77% vs 1.90%, χ² p = 2.8e-10), and false negatives cluster: 177
receptors carry at least one, and 61 of those also have true positives, so the model is
not simply blind to them.

`04` asks whether identical receptor barcodes imply identical perception. 272 molecules
fall into 65 groups sharing a barcode; 389 are unique and 93 activate nothing. Molecules
with the same barcode do share more odour descriptors than random pairs, but the effect
is modest and survives controlling for structural similarity (p = 0.002 among
structurally dissimilar pairs). Read it as a weak signal, not a demonstration.

## Outputs

```
Figures/           01-03: activation heatmaps, PCoA, enrichment bars, chemistry, confusion matrix
Figures_barcodes/  04
Tables_barcodes/   04: barcode groups, shared descriptors, structure control
metadata/          exp_pred_common_pairs.csv - written by 01, read by 03
```

`metadata/exp_pred_common_pairs.csv` holds the 23,782 measured pairs with their predicted
probability and confusion-matrix category, called at 0.9150. Re-run `01` if you change
the threshold; `03` will otherwise disagree with it.

## Inputs

In the repository:

- `Model/Data_Preparation/processed/m2or_pairs_model.csv` — measured pairs
- `Phylogenetic_Analysis/data/ReferenceTree/PF13853.*.fa` — sequences, for the join
- `Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv` — class I / II assignment
- `Phylogenetic_Analysis/data/HumanTree/human433_MFP_reorder.tree` — phylogenetic row order in `02`
- `predictions/aggregated/ASR_Predictions_RefTree/inchi_smiles_odors.csv` — odour descriptors

From the Zenodo record — see `../../README.md` for the download step:

- `predictions/aggregated/Reference_Tree_Predictions/reference_median_probability_wide.csv`

**Known gap.** The final section of `03`, which compares false-positive rate against
patristic distance, reads a precomputed branch-distance matrix that is not yet in the
repository. That section will not run on a fresh clone; everything above it will. The
rest of `03` depends only on `metadata/exp_pred_common_pairs.csv`.
