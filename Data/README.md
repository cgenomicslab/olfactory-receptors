# Data

The measured receptor–odorant bioassays everything else is built on, as downloaded, plus an
exploratory notebook.

```
m2or_pairs.csv   the M2OR export, unmodified
data.ipynb       exploratory data analysis -> figures/
figures/         11 EDA figures (png + svg)
```

## `m2or_pairs.csv`

M2OR (Molecule to Olfactory Receptor), a curated database of published olfactory receptor
deorphanisation assays. **53,444 rows**, one per (receptor, molecule, assay) observation.

> **Separator is `;`, not `,`.** Several fields contain commas.
> `pd.read_csv("m2or_pairs.csv", sep=";")`

| column | meaning |
|---|---|
| `Species` | source organism — 42,917 human, 10,243 mouse, the rest primate/bovine |
| `Gene Name` | receptor gene symbol, e.g. `OR10S1` |
| `UniProt ID` | receptor accession. **Not unique per sequence** — mutants share the parent accession |
| `Mutation` | mutation string, empty for wild type |
| `Sequence` | the amino-acid sequence actually assayed. **This is the real join key** |
| `Molecule Name`, `CID`, `CAS` | odorant identifiers |
| `InChIKey`, `SMILES` | odorant structure |
| `Mixture` | `mono` (41,194), `sum of isomers` (10,981), `mixture` (1,269) |
| `Responsive` | **the label** — 1 if the receptor responded, 0 if not. 3,124 positives (5.8%) |
| `Data Quality` | `primaryScreening` (39,094), `secondaryScreening` (8,478), `ec50` (5,872) |
| `Number of Unique Value Screen` | replicate count behind the row |

**Join on `Sequence`, never on `UniProt ID`.** 17,009 of the modelling rows are mutants,
and they share their parent's accession. Joining on accession folds mutant measurements
onto wild-type predictions; joining on exact sequence excludes them automatically. Every
analysis in this repository does the latter.

**`Responsive` is maxed over replicates.** The same (sequence, odorant) pair can appear
several times at different assay qualities. Where the analyses need one label per pair they
take the maximum, so a single positive replicate makes the pair positive.

## Where it goes next

`Model/Data_Preparation/notebooks/extract_pairs.ipynb` turns this file into the modelling
table `Model/Data_Preparation/processed/m2or_pairs_model.csv` — see
[`../Model/README.md`](../Model/README.md).

## Source

M2OR: <https://m2or.chemsensim.fr/>. Redistributed here so the analysis is reproducible
from the clone; cite the original database, not this copy.
