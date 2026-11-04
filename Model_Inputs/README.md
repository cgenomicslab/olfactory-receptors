# Model inputs

PLinter requires two main data assets for training and inference: the processed table of
receptor–odorant pairs and their precomputed embeddings.

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

Proteins: ESM-C 300M, one `.npy` per protein under `proteins/`, shaped (L + 2, 960) with the
start and end tokens: 722 class A GPCRs, 1,399 M2OR proteins, 584 reference-tree proteins.
Molecules: `molecules/molformer_smiles_embeddings.pt`, 768-d, keyed by SMILES.

Regenerating needs a GPU and the `esm` package. Download them instead:

```bash
python scripts/download_zenodo_data.py --embeddings        # class A and reference-tree arrays, 1.5 GB
python scripts/download_zenodo_data.py --embeddings-full   # also the M2OR proteins and the molecules
```

The `extract_embeddings_*.ipynb` notebooks are how they were made, and draw the Figure 1 maps.
