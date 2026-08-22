# Odorant chemical space against natural products

Where odour chemistry sits inside natural-product chemical space, measured on
**COCONUT 08-2026**. Method follows A. Pittis's `02.Analysis/COCONUT_Odorants/` so the two
runs are directly comparable; the code is reimplemented here so this repository is
self-contained.

## The question

The earlier ChEMBL contrast found odorants sit apart from drug-like chemistry — small,
volatile, nitrogen-free, oxygen-rich. That result is *relational*: it measures the distance
between odour chemistry and **drug** chemistry. Swapping the background for COCONUT removes
exactly the contrast that produced it, because natural products are themselves
nitrogen-poor and oxygen-rich.

> Is odour chemistry genuinely special, or is "the odorant island" just "odorants are the
> volatile natural products, and ChEMBL is neither drugs nor volatiles"?

**Q1 alone is not reportable.** A high kNN enrichment against a natural-product background
can mean odour chemistry is distinctive, or merely that odorants occupy the small volatile
corner of natural-product space. Only the matched control (**Q2**) separates those, and the
two numbers must always travel together. `s06` then puts intervals and nulls on both.

---

## Run order

```bash
conda activate olfactory-receptors     # see ../../environment.yml
cd Chemicals/chemspace
mkdir -p data results

# COCONUT 08-2026, ~236 MB zipped / 668 MB unzipped. Not redistributed here.
curl -o data/coconut_csv-08-2026.zip \
  https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip
unzip -d data data/coconut_csv-08-2026.zip

python s01_build_universe.py --nproc 48   # CPU  ~30 min  (3 min on 64 cores)
python s02_embed.py                       # GPU  ~1 h on a T1000   <-- the only GPU step
python s03_umap.py                        # CPU  ~30 min
python s04_analyze.py                     # CPU  ~20 min   the four questions
python s06_robustness.py                  # CPU  ~10 min   intervals, nulls, leakage
python s05_figures.py                     # CPU  ~2 min    the figures
```

`s01` caches `results/coconut_parents.parquet`, so re-running it is cheap after the first
pass. `s02` refuses to start without a visible GPU rather than silently falling back to a
CPU run that would take days. `s03` skips work whose output already exists. `s05` reads
whatever `s04` and `s06` have written and skips the figures whose inputs are missing, so it
is safe to run at any point.

The optional `s07`–`s11` scripts add the organism-source analyses and the assembled paper
figures; they are not part of the core argument and are documented in their own docstrings.

### Options worth knowing

| script | flag | effect |
|---|---|---|
| `s01` | `--nproc N` | worker processes for descriptor computation (default 48) |
| | `--n-background N` | subsample the background; `0` (default) keeps all of COCONUT |
| `s03` | `--raw` | run UMAP on the full 768-d instead of PCA-50 |
| `s04` | — | no flags; reads `results/`, writes `results/` |
| `s06` | `--seeds N` | matching seeds for the stability check (default 5) |
| | `--null-sets N` | replicates per null (default 5) |
| | `--quick` | 2 seeds, 2 nulls — smoke test, ~3 min |
| `s05` | `--only 01346` | which figures to draw |
| | `--map raw\|pca` | which UMAP layout to draw on |

---

## What each step produces

| step | output | size |
|---|---|---|
| `s01` | `results/coconut_parents.parquet` desalted COCONUT + annotations | ~85 MB |
| | `results/universe.parquet` background + odorants, 22 descriptors, 16 FGs, Joback Tb | ~105 MB |
| `s02` | `results/embeddings.npy` float32 (n, 768) | **~2.2 GB** |
| | `results/embed_index.csv` inchikey, token_len — the join key for everything | ~22 MB |
| `s03` | `results/pca50.npy`, `results/umap.npy` | ~150 MB |
| `s04` | `knn_enrichment.csv` (Q1), `coconut_matched_smd.csv`, `coconut_matched_auc.csv`, `coconut_lda_structure.csv`, `compare_chembl_coconut.csv` (Q2), `pathway_enrichment.csv` (Q3), `fg_enrichment.csv` (Q4, raw **and** matched), `analyze_report.txt` | small |
| `s06` | `robust_matched_auc.csv`, `robust_scaffold_auc.csv`, `robust_knn_null.csv`, `robust_enrichment_ci.csv`, `robustness_report.txt` | small |
| `s05` | `figures/F0_pca_variance`, `F1_np_map`, `F3_what_an_odorant_is`, `F4_two_islands`, `F6_property_maps`, `F7_organism_kingdom` (png + svg) | small |

**Nothing large in `results/` is committed.** `.gitignore` covers the COCONUT download,
`results/*.npy`, `results/*.parquet` and `results/embed_index.csv`. The small CSVs, both
reports, `figures/`, and the two committed inputs in `data/` are kept.

---

## The four questions, and what they are for

### Q1 — are odorants near each other? (`s04`)

For each odorant, what share of its 50 nearest neighbours in the 768-d MolFormer space are
also odorants, divided by the share of the universe that is odorants.

```
odorant fraction among an odorant's 50 NN   0.3849
base rate                                   0.0082
ENRICHMENT                                  46.9x
```

On its own this number means nothing. Read it with Q2.

### Q2 — the matched control (`s04`)

Each odorant is paired with the most similar **non-odorant natural product**, matched on
the three transport properties, then the two groups are compared on equal terms.

| variable | value | meaning |
|---|---|---|
| `MATCH_COVARIATES` | `Tb_joback`, `MolWt`, `LogP` | volatility, size, lipophilicity — the axes on which "odorants are special" is least interesting |
| `CALIPER_SD` | `0.25` | largest allowed distance to a partner, per covariate, in SDs |
| `BALANCE_TOLERANCE` | `0.01` | matching is only accepted if every post-match \|SMD\| is under this |
| `FOLDS` | `5` | cross-validation folds |

Matching moves the covariates from very different to indistinguishable:

```
              SMD before    SMD after
Tb_joback        -2.323      -0.0035
MolWt            -1.765      +0.0014
LogP             -0.437      +0.0021
matched 5,939 of 5,962 odorants (99.9%)
```

Then, how separable are the two groups?

```
features            unmatched   matched     drop
volatility_only         0.973     0.494    0.478     <- collapses to chance, as it must
descriptors_22          0.981     0.667    0.314
molformer_768           0.983     0.795    0.188
```

`volatility_only` landing at 0.494 is the check that matching removed what it was meant to
remove. **If it does not land near 0.50, matching failed and every other AUC is
meaningless** — check the `|SMD|` lines first.

The remaining 0.795 is the part of "odorants are distinctive" that transport properties do
*not* explain. That is the reportable claim.

Q2 also writes the discriminant axis and compares it against the ChEMBL one
(`data/chembl_lda_structure.csv`): Pearson r = **+0.714** over 22 descriptors, with 6
changing sign between backgrounds. The axis is similar but not the same, which is the point
of running two backgrounds.

### Q3 — biosynthetic pathway (`s04`)

Odorants are overwhelmingly fatty-acid derived (49.6% vs 12.2%, OR 7.08) and depleted in
alkaloids (OR 0.31) and polyketides (OR 0.35).

### Q4 — functional groups (`s04`)

Reported twice: raw against all of COCONUT, and matched against the Q2 control. **Quote the
matched column.** Six of sixteen groups reverse direction between them:

```
                     OR raw   OR matched   verdict
fg_thiol              23.90         3.07   survives
fg_aldehyde            3.62         1.18   survives
fg_ester               0.74         1.68   REVERSES
fg_alcohol             0.20         1.14   REVERSES
fg_amide               0.05         0.51   survives
```

A group that reverses was reporting that odorants are small and volatile, not that odour
chemistry favours it.

---

## Robustness — why `s06` is not optional

`s04` produces numbers; `s06` produces the reasons to believe them. It attacks the three
places the analysis could be fooling itself.

**1. Did the matched control depend on one lucky pairing?** The matcher walks odorants in a
random order, so a different seed gives a different control group. `s06` rebuilds the whole
of Q2 under several seeds. The spread is tiny (±0.001–0.007 AUC), so the result is a
property of the chemistry, not of the matcher.

**2. Are the AUCs inflated by near-duplicate molecules?** Random cross-validation lets two
molecules differing by one methyl group land on opposite sides of the split, and a
classifier that memorised one will "predict" the other. `s06` re-scores everything with a
**Bemis–Murcko scaffold split**, where every molecule sharing a core stays on the same side:

```
features            random CV        scaffold
volatility_only   0.491 +/-0.004   0.472 +/-0.007
descriptors_22    0.667 +/-0.000   0.658 +/-0.001
molformer_768     0.794 +/-0.001   0.772 +/-0.004
```

The optimism is small (~0.02), so the AUCs survive. **Quote the scaffold column.**

**3. Is the kNN enrichment big, or only big-looking?** 46.9× has no scale until you know
what a non-answer looks like. Two nulls give it one:

```
observed (odorants)         46.9x
matched null                18.5x  +/- 0.1     <- a set chosen on transport properties alone
random null                  1.0x  +/- 0.0     <- the sanity check that the measure works
observed / matched null      2.53x             <- THE NUMBER TO QUOTE
```

**The 46.9× must never be quoted on its own.** A set of natural products chosen only to
match odorants on size and volatility already clusters at 18.5×. Reporting 46.9× as an
olfactory result would be claiming the transport effect as an olfactory one. The real
signal is the factor of ~2.5 above that null.

The bootstrap interval on the enrichment is [46.1, 47.6] — narrow, because n is large.
Narrow is not the same as correct: it describes sampling noise only, never the matched null.

### The checklist before quoting anything

From `results/robustness_report.txt`:

| quantity | expected |
|---|---|
| matched control, all `\|SMD\|` | < 0.01 — if not, matching failed and every AUC is meaningless |
| volatility (Tb) alone, matched | ~0.50, i.e. chance |
| 22 RDKit descriptors, scaffold split | ~0.66 |
| MolFormer embedding, scaffold split | ~0.77 |
| kNN enrichment, odorants | ~47× |
| kNN enrichment, matched NPs | ~18.5× — **the null that matters** |
| kNN enrichment, random NP sets | ~1× |

---

## Design decisions worth knowing

**Joins are on an InChIKey recomputed by RDKit from SMILES**, never on a COCONUT identifier
or anyone else's ID. A release bump cannot silently break them.

**The odorant set is release-independent.** 5,962 molecules (M2OR 754 + Leffingwell 3,522 +
GoodScents 4,492, deduplicated on InChIKey) built from M2OR and pyrfume, not from COCONUT,
so it transfers across releases unchanged. Committed at `data/reference_sets.csv`.

**Odorants are excluded from the background pool**, so `in_background` is an unbiased sample
of *non-odorant* natural products rather than a mixture.

**The MolFormer revision is pinned** (`7b12d946…`) and everything is fp32. Unpinned,
HuggingFace can move the checkpoint and every number changes silently; fp16 rounding moves
neighbour rankings at the tail, which is exactly where the kNN result lives.

**UMAP is illustrative only.** Every quantitative claim is measured in raw 768-d cosine
space in `s04` and `s06`, never from map coordinates. The map runs on PCA-50 because
NN-descent there is ~15× cheaper and the global layout is visually equivalent; `--raw` does
it on the full 768-d.

**The MolFormer embedding is strongly anisotropic** — the mean vector is 0.79 of the mean
embedding norm, and two random molecules already sit at cosine 0.62. Centring does not
change the UMAP layout and moves the kNN enrichment only from 46.9× to 48.5×, so nothing
depends on it; the diagnostic belongs in Methods because a referee who knows MolFormer will
ask.

---

## Caveats that belong in Methods

- **Curation confound.** COCONUT and the odour lists are human-curated with different
  inclusion criteria. "Odorants are natural-product-like" is partly "both lists were
  assembled by people interested in plants". The matched control mitigates this but cannot
  remove it.
- **Joback is an estimate**, applied well outside its fitting domain for many molecules.
  Every volatility conclusion rests on it — and `thermo` returns NaN silently when absent,
  so check it before trusting a run (`environment.yml` pins it, and
  `scripts/check_reproducibility.py` flags it).
- **The 07-2026 and 08-2026 COCONUT downloads are the same data.** Verified directly:
  identical 738,827 identifiers, identical SMILES for every one, differing only in row
  order. So this rebuild is a reproducibility check, not a release comparison. Whether a
  genuinely *different* release moves the numbers is still untested.
- **M2OR is a mildly biased draw** within the odorant set (AUC 0.73 against other odorant
  lists, skewed smaller).
- **Matching cannot equalise what it does not measure.** The control equalises boiling
  point, weight and logP. Anything correlated with being an odorant but orthogonal to those
  three is still free to drive the residual 0.77 AUC.
