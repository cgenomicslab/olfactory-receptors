# Chemicals

The odorant side: what the molecules are, what they smell like, and where odour chemistry
sits relative to other natural products.

The analysis is in [`chemspace/`](chemspace/) — start with
[`chemspace/figure4_chemical_space.ipynb`](chemspace/figure4_chemical_space.ipynb).

## `odor_datasets/` — the committed tables

| File | Is |
|---|---|
| `inchi_odors_smiles.csv` | **754 molecules** with their odour descriptors. Read by every notebook that asks what something smells like |
| `ligand_pathways.csv` | NPClassifier biosynthetic pathway per molecule, joined from COCONUT |
| `leffingwell_goodscents_smiles.csv` | the wider odorant universe, feeding the 5,962-molecule set used by `chemspace/` |

Descriptors are free text, lowercased and split on commas. There is no controlled
vocabulary — a limitation of the source lists, not a choice here.

[`smiles_to_odors_pyrfume.ipynb`](smiles_to_odors_pyrfume.ipynb) rebuilds these from
pyrfume.

## Two molecule sets, easy to confuse

| Set | Size | Used for |
|---|---|---|
| **M2OR odorants** | **754** | every prediction matrix column; all of `Downstream_Analysis/` |
| **the odorant universe** | **5,962** | `chemspace/` only — M2OR + Leffingwell + GoodScents, deduplicated |

The 754 are the molecules with measured receptor data. The 5,962 are everything known to
smell of anything, used for a chemistry question that needs no receptor measurements.
