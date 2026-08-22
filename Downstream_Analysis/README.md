# Downstream analysis

The notebooks that read the predicted receptor × odorant probability matrices and ask what
they say — about the extant human receptors, and about the reconstructed ancestors.

```
notebooks/reference_tree/   the 433 extant human receptors        (01 -> 05)
notebooks/ancestral/        the 432 reconstructed nodes           (01 -> 06)
scripts/                    shared code the notebooks import
predictions/                the probability matrices (from Zenodo, not Git)
```

Each notebooks folder has its own README with the per-notebook detail. This file covers
what they share: where the data comes from, how a probability becomes a binding call, and
what the shared functions return.

---

## Before you start

```bash
conda activate olfactory-receptors
python scripts/download_zenodo_data.py       # from the repository root
python scripts/check_reproducibility.py      # confirms the clone is complete
```

Then run the notebooks **in numbered order within a folder**. The numbering is a real
dependency, not a suggestion: `reference_tree/01` writes
`metadata/exp_pred_common_pairs.csv`, which `03` reads.

---

## Binarisation: how a probability becomes a binding call

This is the single most important thing to understand before reading any figure, because
every count in every notebook depends on it.

The model emits a probability per pair. Deciding what counts as "binds" needs a rule, and
this project uses **two different rules for two different jobs**. They are not
interchangeable and mixing them up will make numbers disagree.

### The density fill — used to build every matrix

Defined once in [`scripts/calibration.py`](scripts/calibration.py) and shared by both
notebook folders. It separates two questions that a plain threshold conflates:

| question | answered by |
|---|---|
| *How many* cells should be on? | isotonic calibration against the 23,782 measured pairs |
| *Which* cells are on? | rank by model score |

1. Isotonic regression maps model score → measured binding frequency, fitted **once** on
   the tested cells only.
2. That map is applied to every untested cell, giving each a calibrated probability of
   being a real binder.
3. Their sum is the expected number of true binders, `N` (linearity of expectation —
   no threshold is involved).
4. The untested cells are ranked by score and the top `N` set to 1. The score of the
   `N`-th cell is the **density cut**: an *output* of the procedure, never a chosen input.
5. Where a measurement exists it is written over the fill. A measured value always beats a
   predicted one.

The ASR and extant matrices are filled **separately**, each from its own score
distribution, using the **same** calibration map:

```
ASR      432 nodes    cut 0.8875   density 2.2623%
extant   433 human    cut 0.8947   density 2.7453%
```

Reconstructed sequences score slightly lower than real ones, and a per-matrix fill absorbs
that instead of letting it starve the ancestral rows. Carrying one cut onto the other
matrix would put a chosen number back into the procedure.

> **Assumption.** At a fixed model score, tested and untested pairs bind at the same rate.
> This is what licenses transporting the map off the tested cells. It is weaker than
> assuming the tested cells are a random sample of the grid — they are not, being enriched
> for well-studied, broadly-tuned receptors — because it conditions on the score. It is not
> assumption-free: a pair can be untested precisely because nobody expected it to bind.

### The rate-matched cut (0.9150) — used only to report

The cut at which the predicted binding proportion equals the measured one on the tested
cells. It is the right number for **per-pair classification performance** — the confusion
matrix and precision/recall in `reference_tree/01` are computed at it — and the wrong tool
for filling a grid, because recall there is ~0.6, so thresholding the full matrix drops
roughly 40% of real binders.

**It builds nothing.** If you see it used to construct a matrix, that is a bug.

### Experimental overlay: on for extant, off for ancestral

`reference_tree/` overlays the 23,782 measurements onto its fill. `ancestral/` does not —
an ancestor cannot have measurements, so overlaying them on the extant side alone would put
a systematic discontinuity on every ancestor-to-tip edge, exactly where `edge_table`
measures change. Both sides stay prediction-versus-prediction.

---

## Shared code

`scripts/` holds what the notebooks import, so the analysis stays in the notebooks and the
plumbing is not copy-pasted between them.

```python
import sys; sys.path.append("../../scripts")
from ancestral_sets import load_ancestral, decay_table, edge_table
```

### `calibration.py`

| function | returns | notes |
|---|---|---|
| `fit_calibration_map(scores, labels)` | fitted `IsotonicRegression` | fit once, reuse; never refit per matrix |
| `reference_calibration_map(prob_path, pairs_path, fasta_path)` | the one shared map | cached per path triple |
| `density_fill(P, iso, tested_mask=None, experimental=None)` | `FillResult` | `tested_mask=None` means "treat every cell as untested", i.e. no overlay |
| `receptor_weighted_prevalence(row_ids, labels)` | `float` | independent check; unweights testing effort |
| `bin_reweighted_prevalence(tested_scores, tested_labels, grid_scores, n_bins=20)` | `float` | independent check; coarse analogue of the isotonic map |

`FillResult` fields:

| field | meaning |
|---|---|
| `matrix` | `int8` array, same shape as the score grid — the calls |
| `cut` | score of the `N`-th ranked cell — **an output** |
| `density` | fraction of the full grid set to 1 |
| `n_expected` | `N`, expected true binders among untested cells |
| `n_experimental_pos` | measured positives written over the fill |
| `orphan_receptors` / `orphan_ligands` | rows / columns summing to 0 |
| `calibrated_mean` | mean calibrated probability over untested cells |

### `ancestral_sets.py`

Module-level paths — `ASR_MATRIX`, `REF_MATRIX`, `PAIRS`, `FASTA`, `ODORS`, `ASR_TREE`,
`CLASSES` — resolve relative to the repository root, so notebooks work from any directory.

| function | returns | key parameters |
|---|---|---|
| `load_ancestral(mode="density", top_fraction=None, threshold=0.5)` | `Ancestral` | `mode`: `"density"` (default), `"rank"`, `"absolute"` |
| `density_grid()` | `(binary, prob, results, ancestral_names)` | cached; both fills run once per session |
| `extant_calibration()` | `(threshold, measured_rate)` | the 0.9150 reporting cut and the 4.7431% rate |
| `decay_table(threshold=None)` | ligand count vs patristic distance, per node | `None` = density fill; a number = absolute cut |
| `edge_table(threshold=None)` | parent→child similarity for all 864 edges | includes a threshold-free weighted Jaccard |
| `node_topology(nodes=None, threshold=None)` | where each node sits, and its gains/losses | `nodes` defaults to `TEN_NODES` |
| `node_sets(nodes=None, threshold=None)` | `{node_number: set_of_SMILES}` | |
| `mpl_times(ingroup="node_1")` / `raw_times(...)` | `Timescale` | the two time axes `06` brackets its answer with |
| `lineage_slice(times, when, ...)` | lineage names crossing relative time `when` | a **standing**, not cumulative, repertoire |
| `trajectory(times, grid=None, ..., blank_above=None)` | one row per time point | `blank_above` drops hyper-broad rows |
| `functional_groups(smiles)` / `descriptors(smiles)` / `odor_tags()` | RDKit and odour-tag tables | |
| `save_fig(fig, name)` / `save_table(frame, name)` | — | writes to `Figures/` (SVG) and `Tables/` (CSV) |

**`mode`, the parameter that changes every number:**

| `mode` | what it does | when to use it |
|---|---|---|
| `"density"` | the shared density fill; cut is an output | **the default — use this** |
| `"absolute"` | calls a ligand bound when probability > `threshold` | only for the threshold sweeps in `ancestral/04` |
| `"rank"` | each node's top `top_fraction` of ligands | equalises repertoire sizes; removes the reconstruction-confidence gradient |

`"absolute"` at the default 0.5 is far more permissive than the rate-matched cut — on
measured extant pairs it gives precision 0.27, so roughly three in four positives are
false. Repertoire sizes under it are not binding counts.

**`Ancestral` fields** returned by `load_ancestral`:

| field | meaning |
|---|---|
| `predictions` | the binarised ASR matrix |
| `node_presence` | SMILES × {common, class1, class2}_ancestor |
| `pattern_df` / `pattern_sets` | each ligand's three-bit pattern, read as `[class1, common, class2]` |
| `code_sets` | named groupings: `class1_side`, `class2_side`, `shared`, `common_only` |
| `sets` | `common`, `class1`, `class2`, plus gains and losses |
| `density_cut` / `density` | the cut that fell out of the fill, and the resulting density |
| `per_node_cut` | probability each node's cut landed on |

The three core nodes map as `node_1_ASR → common_ancestor`, `node_2_ASR → class1_ancestor`,
`node_3_ASR → class2_ancestor`.

### `plotting_functions.py`

The rcParams every notebook imports, so figures match across folders.

---

## Getting the data

Prediction outputs are not in Git — they live in the project's Zenodo record alongside the
embeddings. The record ID and per-asset SHA-256 checksums are in `zenodo_manifest.json` at
the repository root.

```bash
python scripts/download_zenodo_data.py          # predictions + pooled embeddings (~50 MB)
python scripts/download_zenodo_data.py --list   # show every asset
python scripts/download_zenodo_data.py --assets all
```

Each archive's checksum is verified before extraction into `predictions/`, which is
gitignored. The dataset carries **probabilities only**, for `ASR_Predictions` and
`Reference_Tree_Predictions`:

```
runs/<Category>/<run files>        per-run probabilities, all five runs
aggregated/<Category>/<matrix>     median-probability wide matrices
model_run_evaluation_metrics.csv   per-run evaluation metrics
MANIFEST.json                      per-file sizes and SHA-256 checksums
```

No binary call table is shipped, deliberately: each notebook applies the rule it is
testing, so a pre-thresholded table would silently fix a choice that belongs downstream.

The per-run files are published so the medians can be reproduced rather than taken on
trust. Regenerating from them yields byte-identical matrices:

```bash
python Downstream_Analysis/scripts/aggregate_prediction_runs.py
python Downstream_Analysis/scripts/aggregate_prediction_runs.py --category ASR_Predictions --dry-run
```

| flag | effect |
|---|---|
| `--category` | one of `ASR_Predictions`, `Reference_Tree_Predictions`, `M2OR_Unseen_Pairs_Predictions`, `all` |
| `--dry-run` | print what would be written, write nothing |
| `--archive-dir` | use a specific `Predictions_M2OR-*` export instead of the latest |
| `--limit-runs` | testing only — process the first N runs per category |

---

## Publishing a Zenodo release

Both halves of the record stage into the gitignored `release/` directory at the repository
root.

```bash
python Downstream_Analysis/scripts/prepare_zenodo_predictions.py --version v1
python Model/Embeddings/prepare_zenodo_embeddings.py --version v1
```

The predictions script reads the per-run files from the latest
`predictions/archive/Predictions_M2OR-*` export, drops each run's `prediction` column (a
0.5-thresholded derivative that would conflict with the rules used in the notebooks),
copies probability values **as text** so published numbers are byte-identical to the model
output, and writes a ZIP plus a `.sha256`.

| flag | effect |
|---|---|
| `--version` | version label used in the archive name |
| `--no-medians` | publish the per-run files alone |
| `--overwrite` | replace an existing ZIP |
| `--pooled-only` *(embeddings)* | skip the multi-GB per-residue archives |
| `--datasets` *(embeddings)* | subset of `gpcr,m2or,reference_tree`, or `all` |

Then: upload every `.zip` and `.sha256` from `release/`, publish, and fill in
`zenodo_record_id` and each asset's `sha256` in `zenodo_manifest.json`. Rebuilding an
archive changes its checksum (ZIPs embed timestamps), so re-sync the manifest if you
regenerate anything after uploading.
