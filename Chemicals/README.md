# Chemicals

The odorant side of the project: what the molecules are, what they smell like, and where
odour chemistry sits relative to other natural products.

Start with [`figure4_chemical_space.ipynb`](figure4_chemical_space.ipynb). It shows Figure 4
and the Supplementary chemical-space figure inline, with the numbers behind every panel. It
reads finished results and does not re-run anything.

## The question

Is odour chemistry genuinely its own region of chemical space, or just the small volatile
corner of natural-product space?

ChEMBL was the first background and was the wrong one: ChEMBL is drug chemistry, so most of
the separation it showed was "odorants are not drugs". COCONUT is the fair background,
because odorants and natural products are already chemically comparable. The differences
between the two sets are the result, not a confound to be controlled away.

## Two molecule sets, easy to confuse

| set | size | used for |
|---|---|---|
| **M2OR odorants** | **754** | every prediction matrix column; all of `Downstream_Analysis/` |
| **the odorant universe** | **5,962** | this folder only |

The 754 are the molecules with measured receptor data. The 5,962 are everything known to
smell of anything: M2OR (754) plus Leffingwell (3,522) plus GoodScents (4,492), deduplicated
on InChIKey.

## The main result

Odorants are **46.9x** more clustered than chance against a COCONUT background. Natural
products matched to odorants on size and volatility already cluster at **18.5x**. The number
to report is the ratio, **2.53x**. Neither number means anything without the other.

For functional groups and pathways, report the **raw** odds ratios. Odorants are small and
volatile because that is what makes a molecule smellable, so matching it away removes the
thing being studied. Six of sixteen groups reverse direction between raw and matched; the
matched column belongs beside the raw one, never instead of it.

## Input files

Four files are committed. Three are downloaded and are not.

| file | size | source |
|---|---|---|
| `data/reference_sets.csv` | 0.5 MB | committed. The 5,962 odorants, keyed by InChIKey and canonical SMILES. Release-independent. |
| `odor_datasets/inchi_odors_smiles.csv` | | committed. 754 molecules with their odour descriptors. |
| `odor_datasets/ligand_pathways.csv` | | committed. NPClassifier pathway per molecule, joined from COCONUT. |
| `data/coconut_csv-08-2026.csv` | 668 MB | downloaded, gitignored |
| `data/nodes.dmp`, `data/names.dmp` | 512 MB | downloaded, gitignored |

Odour descriptors are free text, lowercased and split on commas. There is no controlled
vocabulary. That is a limit of the source lists, not a choice made here.

### Downloading them

```bash
python 00_download_inputs.py                  # both
python 00_download_inputs.py --only coconut   # just COCONUT
```

It skips anything already extracted, and deletes the `.zip`/`.tar.gz` afterwards unless
`--keep-archives` is given.

## The pipeline

The numbering is a reading order, not a dependency chain. There are two branches off `01`,
and only one of them needs a GPU.

```
01_build_universe      COCONUT + reference_sets.csv          ->  universe.parquet
   |                                                             730,571 molecules
   |
   +-- needs a GPU ----------------------------------------------------------------
   |   02_embed           MoLFormer embeddings                ->  embeddings.npy
   |   03_umap            PCA-50, then UMAP                   ->  pca50.npy, umap.npy
   |   04_analyze         Q1-Q4: clustering, matched control,
   |                      pathways, functional groups         ->  *_enrichment.csv
   |   06_robustness      seeds, scaffold split, kNN nulls    ->  robust_*.csv
   |
   +-- no GPU needed --------------------------------------------------------------
       07_organism_source which organisms make odorants       ->  organism_*.csv
       08_descriptor_scan sweeps COCONUT's other 34 columns   ->  scan_*.csv
       09_source_breadth  tests the "more organisms" claim    ->  source_breadth.csv
       10_m2or_organisms  the same, for one panel at a time   ->  m2or_*.csv
       12_aquatic_sources habitat of each source organism     ->  aquatic_*.csv

05_figures      working figures, one per question             ->  figures/F*.svg
11_paper_figures  Figure 4 and the Supplementary, at print size
```

The whole organism branch runs without a GPU. If you only want the source-organism results,
run `01`, then `07`, then `10` and `12`.

### What each step does

**`01_build_universe`** reads 10 of COCONUT's 44 columns, desalts every SMILES to its parent
with RDKit and recomputes the InChIKey. Both sides of the join are canonicalised the same
way, so the join never depends on a COCONUT identifier and a release bump cannot silently
break it. Odorants are removed from the background, so "background" means natural products
that are not odorants. Then 22 descriptors, 16 SMARTS functional groups and a Joback boiling
point are computed for every molecule.

**`02_embed`** is the only step that needs a GPU, and it exits rather than running on CPU.
The model is `ibm-research/MoLFormer-XL-both-10pct`, pinned to a commit hash. MoLFormer's
`max_position_embeddings` is 202, so molecules whose SMILES tokenises longer than that cannot
be encoded and are dropped: 4,164 molecules, 0.57%, **none of them odorants**. Batches are sorted by length, which is
worth about 3x.

**`03_umap`** reduces 768 dimensions to 50 with PCA, then runs UMAP on that. The seed is
fixed, which costs parallelism because UMAP runs single-core when seeded. The map is for
looking at; every distance in the analysis is measured in the full 768 dimensions.

**`04_analyze`** asks four questions. Q1: are odorants near each other (50 nearest
neighbours, cosine, full 768-d)? Q2: does anything survive once size and volatility are
equalised? Q3: which biosynthetic pathways? Q4: which functional groups? Fisher exact tests
with Benjamini-Hochberg FDR.

**`06_robustness`** attacks Q1 and Q2 three ways: rebuilding the matched control under
several seeds, re-estimating every AUC under a Bemis-Murcko scaffold split instead of random
cross-validation, and putting the kNN enrichment beside two nulls.

**`07`-`12`** work from COCONUT's source-organism field, resolved against NCBI taxonomy in
three passes (full name, genus species, genus alone; about 94% resolve). `09` is a **negative
result**: it tests whether odorants really occur in more organisms, and finds the effect
collapses once database membership is controlled for. Do not report it as a finding.

## Outputs

`results/` holds three kinds of file.

**Large intermediates.** Gitignored, on Zenodo, regenerable only with a GPU:
`embeddings.npy` (2.1 GB), `pca50.npy`, `universe.parquet`, `coconut_parents.parquet`,
`embed_index.csv`, `umap.npy`, `umap_raw768.npy`, `aquatic_sources.csv`.

```bash
python scripts/download_zenodo_data.py --assets chemical_space
```

**Small results.** Committed. These are what the figures and the supplementary tables read:
`fg_enrichment.csv`, `pathway_enrichment.csv`, `knn_enrichment.csv`,
`organism_rank_enrichment.csv`, `organism_kingdom_summary.csv`, `robust_*.csv`,
`coconut_matched_*.csv`, `aquatic_sources_panel.csv`, `m2or_*.csv`.

**Reports.** Committed, plain text, meant to be read: `analyze_report.txt`,
`robustness_report.txt`, `organism_report.txt`, `scan_report.txt`, `source_breadth.txt`,
`aquatic_sources_report.txt`.

`figures/` holds the working figures (`F0`-`F7`, `S1`, `X1`-`X3`) and the two assembled ones,
`Figure4_chemical_space` and `SupplementaryFigure_chemical_space`, drawn at final print size.
[`FIGURE_LEGENDS.md`](figures/FIGURE_LEGENDS.md) describes each.

## Running it

```bash
conda env create -f environment.yml && conda activate olfactory-receptors
```

```bash
python 01_build_universe.py --nproc 48   # CPU  ~30 min (3 min on 64 cores)
python 02_embed.py                       # GPU  ~1 h on a T1000
python 03_umap.py                        # CPU  ~30 min
python 04_analyze.py                     # CPU  ~20 min
python 06_robustness.py                  # CPU  ~10 min  (--quick for a ~3 min test)
python 05_figures.py                     # CPU  ~2 min
```

`06` runs before `05` on purpose: `05` draws figures that need `06`'s output. `01` caches
`coconut_parents.parquet`, `03` skips work whose output exists, and `05` skips any figure
whose inputs are missing, so all three are cheap to re-run.

## Checking a run came out right

From `results/analyze_report.txt`:

| quantity | expected |
|---|---|
| matched control, worst `\|SMD\|` | < 0.01, or matching failed and every AUC below it is meaningless |
| odorants matched | ~99.9% of 5,962 |
| volatility alone, matched | ~0.49, i.e. chance |
| 22 descriptors, matched | ~0.67 |
| MolFormer embedding, matched | ~0.79 |

From `results/robustness_report.txt`:

| quantity | expected |
|---|---|
| volatility alone, scaffold split | ~0.47 |
| 22 descriptors, scaffold split | ~0.66 |
| MolFormer embedding, scaffold split | ~0.77 |
| seed-to-seed spread | < 0.01 AUC |
| kNN enrichment, odorants | ~47x |
| kNN enrichment, matched | ~18.5x |
| kNN enrichment, random | ~1x |

Two results mean something is wrong: an embedding AUC near 0.9 means the matching failed,
so check the `|SMD|` lines first; a random null that is not ~1x means the neighbour search
or the labelling is wrong, and nothing else in the file can be trusted.

## Limits

1. **Joback boiling points are estimates**, fitted mainly on small molecules. On the odorant
   side they are plausible. On the background side they are not: the median predicted
   boiling point is 819 C, and `Tb_joback` correlates with `MolWt` at rho 0.93 there. Treat
   the matched control as matched mostly on size, and make no absolute volatility claim
   about the background.
2. **COCONUT records where people have looked.** Plants are over-studied for volatiles and
   *Streptomyces* for antibiotics, so the organism comparison is partly a comparison between
   two literatures.
3. **The background is "not on an odorant list"**, not "verified odourless". It certainly
   contains untested odorous compounds.
