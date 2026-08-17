# Olfactory receptors

Olfactory receptor–ligand interactions.

![Project schema](olfaction_schema.png)

## Layout

```
Chemicals/               odorant datasets, ChEMBL, SMILES to odour-descriptor mapping
Data/                    M2OR receptor-odorant bioassay pairs
Model/
  Data_Preparation/      building the training pairs
  Embeddings/            ESM-C protein and MolFormer molecule embeddings
Phylogenetic_Analysis/   trees: class A GPCRs, six species, the 433 human ORs
Downstream_Analysis/     what the predictions say  -> start here
  notebooks/reference_tree/   extant human receptors
  notebooks/ancestral/        reconstructed ancestral nodes
scripts/                 data download
```

Each analysis folder has its own README. `Downstream_Analysis/README.md` is the one to read
first.
