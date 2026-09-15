# Model inputs

What the binding model reads: the modelling table, and the embeddings. **The model itself is
not here** — its architecture and training code live in a separate repository.

## `Data_Preparation/`

[`extract_pairs.ipynb`](Data_Preparation/extract_pairs.ipynb) turns `Data/m2or_pairs.csv`
into `processed/m2or_pairs_model.csv` — 52,175 rows, 3,066 positive.

| Column | Is |
|---|---|
| `Sequence` | the assayed amino-acid sequence — the join key everywhere downstream |
| `Responsive` | the label, 0/1 |
| `seq_id` | receptor id, 1,399 unique |
| `smiles_id` | molecule id, 754 unique |
| `Class` | class 1 or class 2 |

`seq_id` and `smiles_id` index the prediction matrices, so this file also maps a matrix
column back to a structure.

## `Embeddings/`

Proteins: ESM-C 300M, one `.npy` each, shaped (L, 960), under `proteins/` — 722 class A
GPCRs, 1,399 M2OR proteins, 584 reference-tree proteins. Molecules:
`molecules/molformer_smiles_embeddings.pt`, 768-d, keyed by SMILES.

Regenerating needs a GPU and the `esm` package. Download them instead:

```bash
python scripts/download_zenodo_data.py                                    # pooled, 10 MB
python scripts/download_zenodo_data.py --assets embeddings_reference_tree # per-residue, 663 MB
```

The `extract_embeddings_*.ipynb` notebooks are how they were made, and draw the Figure 1 maps.

> The class A set drops BOS/EOS before pooling; the reference-tree set does not. Each map is
> reproduced under the convention it was published with.

To package a release: `python Model_Inputs/Embeddings/prepare_zenodo_embeddings.py --version v1`,
output to the gitignored `release/`.
