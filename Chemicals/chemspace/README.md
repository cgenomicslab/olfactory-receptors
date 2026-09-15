# Chemical space

Is odour chemistry genuinely distinctive, or just the small volatile corner of
natural-product space?

Start with [`figure4_chemical_space.ipynb`](figure4_chemical_space.ipynb). It shows
Figure 4 and Extended Data Figure 7 inline, with the numbers behind every panel, and does
not re-run the pipeline.

## The one result to read carefully

Odorants are **46.9x** more clustered than chance against a COCONUT background. But natural
products matched to odorants **on size and volatility alone** already cluster at **18.5x**.

The reportable signal is the ratio, about **2.5x** — not the 46.9x. Neither number should
be quoted without the other.

## Running the pipeline

Scripts run in order. Each writes to `results/` and is safe to re-run.

| | Does | Needs |
|---|---|---|
| `s01_build_universe.py` | build the 726,407-molecule universe from COCONUT + odorant lists | 700 MB download |
| `s02_embed.py` | MolFormer embeddings for all of them | **GPU** |
| `s03_umap.py` | the 2-D projection | |
| `s04_analyze.py` | clustering, enrichment, the matched control | |
| `s05_figures.py` | working figures | |
| `s06_robustness.py` | nulls and the scaffold split | |
| `s07`–`s10` | organism sources, descriptor scan, source breadth | |
| `s11_paper_figures.py` | **Figure 4 and Extended Data Figure 7** | |

```bash
python Chemicals/chemspace/s11_paper_figures.py     # rebuild just the paper figures
```

**`s02` needs a GPU and refuses to run on CPU.** Its output is too large to archive, so
without one this analysis cannot be reproduced from scratch. That is a deliberate limit.
See [`RUN_ON_GPU.md`](RUN_ON_GPU.md) for the environment traps.

To get the finished intermediates instead of recomputing them:

```bash
python scripts/download_zenodo_data.py --assets chemical_space
```

## Three limits to carry into any reading

1. **Joback boiling points are estimates**, fitted mainly on small molecules. The odorant
   side is sound; absolute volatility claims about the background are not.
2. **COCONUT records where people have looked.** Plants are over-studied for volatiles and
   *Streptomyces* for antibiotics, so the organism comparison is partly between two
   literatures.
3. **The background is "not on an odorant list"**, not "verified odourless". It certainly
   contains untested odorous compounds.
