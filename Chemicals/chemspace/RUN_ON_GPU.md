# Running the chemical-space pipeline on a GPU box

Everything except `s02` is CPU-only. `s02` needs a visible GPU and refuses to start without
one rather than silently falling back to a run that would take days.

## Environment

The pipeline needs, in one interpreter: `rdkit`, `torch` (CUDA), `transformers`, `umap`,
`sklearn`, `pyarrow`, `pandas`, `matplotlib`, `scipy` and `thermo`.

Two traps, both of which fail quietly rather than loudly:

- **`thermo` is the one dependency nobody has.** Without it `tb_joback` returns `nan` for
  every molecule and no error is raised, so the volatility half of the analysis silently
  becomes a column of NaNs. Check it before starting.
- **`transformers` must be 4.x if `torch` is below 2.5.** transformers 5.x disables the
  PyTorch backend when it finds an older torch, printing an easily-missed warning
  (`Disabling PyTorch because PyTorch >= 2.5 is required`) and then failing in `s02` when
  the model cannot be loaded. MoLFormer's `trust_remote_code` implementation targets 4.x
  in any case.

```bash
python -c "from thermo import Joback; print('ok')"          # must print ok
python -c "import torch; print(torch.cuda.is_available())"  # must print True
python -c "import transformers; print(transformers.__version__)"
python -c "import torch, transformers; from transformers import AutoModel; print('stack ok')"
```

`nvidia-smi` failing with an NVML version mismatch does not mean the GPU is unusable — the
CUDA runtime and the driver are versioned separately. Trust `torch.cuda.is_available()` and
a real allocation over `nvidia-smi`.

## Inputs

Two small files live in `data/` and are committed, because the analysis cannot be reproduced
without them:

| file | what it is |
|---|---|
| `data/reference_sets.csv` | the 5,962-molecule odorant list (M2OR + Leffingwell + GoodScents), keyed by InChIKey and canonical SMILES. Release-independent. |
| `data/chembl_lda_structure.csv` | the ChEMBL discriminant axis, for the two-background comparison in `s04`. |

The COCONUT dump itself is ~700 MB and is **not** committed. Download it once:

```bash
cd Chemicals/chemspace
mkdir -p data results
curl -o data/coconut_csv-08-2026.zip \
  https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip
unzip -d data data/coconut_csv-08-2026.zip
```

## Run

```bash
python s01_build_universe.py --nproc 48   # CPU  ~30 min (3 min on 64 cores)
python s02_embed.py                       # GPU  ~1 h on a T1000   <-- the only GPU step
python s03_umap.py                        # CPU  ~30 min
python s04_analyze.py                     # CPU  ~20 min
python s06_robustness.py                  # CPU  ~10 min  (--quick for a ~3 min smoke test)
python s05_figures.py                     # CPU  ~2 min
```

`s06` uses a GPU when one is visible and is perfectly happy without one: its cost is a
handful of exact kNN searches and some greedy matching, both of which run in a couple of
minutes each on 16 CPU cores. Only `s02` genuinely needs the GPU.

`s01` caches `results/coconut_parents.parquet`, so re-running it is cheap after the first
pass. `s03` skips work whose output already exists. `s05` skips any figure whose inputs are
missing, so it is safe to run at any point.

## Checking the result

`s04` alone is not enough to quote. Check these before using any number.

From `results/analyze_report.txt` (Q2, random cross-validation):

| quantity | expected |
|---|---|
| matched control, all `\|SMD\|` | < 0.01 — if not, matching failed and every AUC below is meaningless |
| odorants matched | ~99.9% of 5,962 |
| volatility (Tb) alone, matched | ~0.49, i.e. chance |
| 22 RDKit descriptors, matched | ~0.67 |
| MolFormer embedding, matched | ~0.79 |

From `results/robustness_report.txt` (`s06`, the numbers to actually report):

| quantity | expected |
|---|---|
| volatility (Tb) alone, scaffold split | ~0.47 |
| 22 RDKit descriptors, scaffold split | ~0.66 |
| MolFormer embedding, scaffold split | ~0.77 |
| seed-to-seed spread, any feature set | < 0.01 AUC |
| kNN enrichment, odorants | ~47x |
| kNN enrichment, matched NPs | ~18.5x — **the null that matters** |
| kNN enrichment, random NP sets | ~1x |
| observed / matched null | ~2.5x |

If the embedding AUC comes out near 0.9, the matching failed — check the `|SMD|` lines first.
If the random null is not ~1x, the neighbour search or the labelling is wrong, and nothing
else in the file can be trusted.

The 47x is only ~2.5x above the matched null, so it must never be quoted on its own. Quote
the scaffold-split AUCs rather than the random-CV ones, and the matched functional-group odds
ratios rather than the raw ones: six of sixteen groups reverse direction between them.

## What not to commit

`results/*.npy`, `results/*.parquet`, `results/embed_index.csv` and the COCONUT download are
gigabytes and are gitignored. The small CSVs, the two reports and `figures/` are kept.
