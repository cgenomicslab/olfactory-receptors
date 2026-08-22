# Model inputs

Everything the binding model consumes: the modelling table built from M2OR, and the protein
and ligand embeddings.

```
Data_Preparation/
  notebooks/extract_pairs.ipynb    M2OR export -> the modelling table
  processed/                       the table, plus the phylogeny FASTAs built alongside it
Embeddings/
  extract_embeddings_*.ipynb       ESM-C 300M protein embeddings
  proteins/                        per-residue (L, 960) arrays, one .npy per protein
  molecules/                       MolFormer ligand embeddings
  prepare_zenodo_embeddings.py     packages the embeddings for release
```

> **The model itself is not in this repository.** This folder covers its inputs; the
> downstream folders cover its outputs. The training code lives separately.
> <!-- TODO: replace with the URL / DOI of the model repository before submission. -->

---

## `Data_Preparation`

`extract_pairs.ipynb` reads `Data/m2or_pairs.csv` and writes
`processed/m2or_pairs_model.csv`: **52,175 rows**, one per (sequence, molecule) observation
that survived filtering.

| column | meaning |
|---|---|
| `Species` | source organism |
| `UniProt ID` | receptor accession — shared by a wild type and its mutants |
| `Sequence` | the assayed amino-acid sequence. The join key everywhere downstream |
| `SMILES` | odorant structure |
| `Mixture` | `mono`, `sum of isomers` or `mixture` |
| `Responsive` | **the label**, 0/1. 3,066 positives (5.9%) |
| `seq_id` | stable receptor ID, `SEQ0000`-style. **1,399 unique sequences** |
| `smiles_id` | stable molecule ID, `SML0000`-style. **754 unique molecules** |
| `mutation` | `True` for a mutant sequence (17,009 rows), `False` for wild type |
| `Class` | `class 1` (9,003) or `class 2` (43,172) receptor |

`seq_id` and `smiles_id` are what the prediction matrices are indexed by, so this file is
also the lookup table joining a matrix column back to a structure:

```python
pairs = pd.read_csv("Model/Data_Preparation/processed/m2or_pairs_model.csv")
smiles_map = pairs[["smiles_id", "SMILES"]].drop_duplicates()
```

`processed/` also holds the FASTAs and trees built from the same sequences for the
phylogenetic work (`m2or_sequences_phylogeny*.fasta`, `*_fasttree.nw`).

---

## `Embeddings`

### Proteins — ESM-C 300M

`extract_embeddings_GPCR.ipynb`, `extract_embeddings_m2or.ipynb` and
`extract_embeddings_reference_tree.ipynb` each run ESM-C 300M over a FASTA and write one
`.npy` per protein, shaped **(L, 960)** where `L` is sequence length.

| directory | contents |
|---|---|
| `proteins/gpcr_esmc_300m_embeddings/` | 722 class A GPCRs |
| `proteins/m2or_esmc_300m_embeddings/` | 1,399 M2OR proteins |
| `proteins/reference_tree_esmc_300m_embeddings/` | 584 reference-tree proteins |
| `proteins/sae_features_6b_human_ancestors/` | 968 sparse-autoencoder feature files |

**The model consumes the mean-pooled form**, not the per-residue arrays: each (L, 960)
matrix is reduced with `emb.mean(axis=0)` to a single 960-vector per protein. Both
representations are published — the pooled matrices are small, the per-residue arrays are
multi-GB — so anyone can verify the reduction rather than trust it.

These need a GPU and the `esm` package to regenerate, which is why they are archived. Most
people should download them instead:

```bash
python scripts/download_zenodo_data.py                                # pooled (10 MB)
python scripts/download_zenodo_data.py --assets embeddings_reference_tree   # per-residue (663 MB)
```

### Molecules — MolFormer

`molecules/molformer_smiles_embeddings.pt` is a torch dict keyed by SMILES, giving a
**768**-dimensional vector per molecule, 754 in total. Indexed by `smiles_id` to join the
prediction matrices.

### SAE feature analysis

`esmc_sae_feature_interpretation.ipynb` and `sae_geatures_exploration.ipynb` explore sparse
autoencoder features over the ESM-C representations, with `proteins/sae_fingerprints_6b.npz`
and `proteins/sae_feature_descriptions.json`. Exploratory — not part of the main result.

---

## Packaging a release

```bash
python Model/Embeddings/prepare_zenodo_embeddings.py --version v1              # everything
python Model/Embeddings/prepare_zenodo_embeddings.py --version v1 --pooled-only  # small archives only
```

| flag | effect |
|---|---|
| `--pooled-only` | skip the multi-GB per-residue archives |
| `--datasets` | subset of `gpcr,m2or,reference_tree`, or `all` |
| `--molecules-only` / `--skip-molecules` | control the MolFormer archive |
| `--overwrite` | replace existing archives |

Output goes to the gitignored `release/` directory at the repository root, alongside the
predictions archive. See [`../Downstream_Analysis/README.md`](../Downstream_Analysis/README.md#publishing-a-zenodo-release).
