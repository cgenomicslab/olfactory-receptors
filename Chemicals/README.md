# Chemicals

The odorant side of the dataset: what the molecules are, what they smell like, and where
odour chemistry sits relative to other chemical space.

```
odor_datasets/        odour descriptors and pathway labels per molecule
smiles_to_odors_pyrfume.ipynb   builds those tables from pyrfume
chemspace/            odorants against a natural-product background   <- the analysis
chembl/               the ChEMBL bulk download (gitignored, ~1.5 GB)
embeddings/           MolFormer embeddings of ChEMBL (gitignored, ~9 GB)
chembl_molecules.ipynb           the earlier ChEMBL contrast
odorant_chemical_space.ipynb    exploratory chemical-space work
```

---

## `odor_datasets/`

The committed tables the downstream notebooks read.

### `inchi_odors_smiles.csv` — the odour descriptor table

**754 molecules**, one per row. Read by `ancestral_sets.odor_tags()` and by every notebook
that asks what something smells like.

| column | meaning |
|---|---|
| `InChIKey` | structure key, recomputed rather than trusted from any database |
| `SMILES` | canonical SMILES — the join key to the prediction matrices via `smiles_id` |
| `Odors` | comma-separated descriptor list, e.g. `herbal, orange, peppery, terpenic, pine` |

Descriptors are free text, lowercased and split on commas when parsed. There is no
controlled vocabulary — that is a limitation of the source lists, not a choice here.

### `ligand_pathways.csv`

NPClassifier biosynthetic pathway labels joined from COCONUT on InChIKey, used by
`reference_tree/05`. 536 of the 754 molecules are in COCONUT and 481 carry a pathway.

### `leffingwell_goodscents_smiles.csv`

The wider odorant universe (Leffingwell 3,522 + GoodScents 4,492) built from pyrfume. Feeds
the 5,962-molecule odorant set used by `chemspace/`.

`smiles_to_odors_pyrfume.ipynb` rebuilds these from pyrfume if you need to.

---

## `chemspace/` — the main chemical-space analysis

Whether odour chemistry is genuinely distinctive, or merely the small volatile corner of
natural-product space. **This is the part with its own full documentation:**
[`chemspace/README.md`](chemspace/README.md), plus
[`chemspace/RUN_ON_GPU.md`](chemspace/RUN_ON_GPU.md) for the environment traps.

The short version: odorants are 46.9× more clustered than chance against a COCONUT
background — but a set of natural products *matched to odorants on size and volatility
alone* already clusters at 18.5×. The reportable signal is the ratio, ~2.5×, not the 46.9×.
Neither number may be quoted without the other.

---

## `chembl/` and `embeddings/` — the earlier contrast

The first version of this analysis used ChEMBL as the background and found odorants sit
apart from drug-like chemistry. That result is *relational*: much of it was "odorants are
not drugs". `chemspace/` replaces the background with natural products precisely to remove
that confound, so **the ChEMBL work is superseded** — kept because the discriminant axis it
produced (`chemspace/data/chembl_lda_structure.csv`) is still used as a comparison in
`s04`'s Q2.

Both directories are gitignored: the ChEMBL bulk download is ~1.5 GB and its MolFormer
embedding array is 8.8 GB. Re-download from ChEMBL if you need them; nothing in the current
analysis does.

`chembl_molecules.ipynb` and `odorant_chemical_space.ipynb` are the notebooks behind that
earlier work. They will not run on a fresh clone without those downloads.

---

## The two molecule vocabularies

Worth keeping straight, because they are different sizes and easy to confuse:

| set | size | where it is used |
|---|---|---|
| **M2OR odorants** | **754** | every prediction matrix column; all of `Downstream_Analysis/` |
| **the odorant universe** | **5,962** | `chemspace/` only — M2OR + Leffingwell + GoodScents, deduplicated on InChIKey |

The 754 are the molecules with measured receptor data. The 5,962 are everything known to
smell of anything, used to ask a chemistry question that does not need receptor
measurements. M2OR is a mildly biased draw from the larger set (AUC 0.73 against the other
lists, skewed smaller) — noted in `chemspace/README.md` as a caveat for Methods.
