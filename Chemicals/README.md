# Chemicals

Places the odorants in natural-product chemical space. Every known odorant is compared with the
whole COCONUT natural-product database: how clustered odorants are, which functional groups and
biosynthetic pathways they carry, and which organisms they are isolated from.

Everything below runs from `Chemicals/`.

## Two odorant sets

| set | molecules | used for |
|---|---|---|
| M2OR ligands | 754 | the receptor predictions and all of `Downstream_Analysis/` |
| all odorants | 5,962 | this folder |

The 5,962 are M2OR (754), Leffingwell (3,522) and GoodScents (4,492), deduplicated on InChIKey.
Both lists are in `processed_data/`:

* `m2or_inchi_odors_smiles.csv`: the 754 M2OR ligands with their odour descriptors
* `odorants_m2or_leffingwell_goodscents.csv`: the 5,962 odorants and which list each comes from
* `ligand_pathways.csv`: the NPClassifier pathway of each M2OR ligand, taken from COCONUT

The first two are written by `c01_odorant_sets.ipynb`. The third is written by `build_pathways()`
in `Downstream_Analysis/notebooks/reference_tree/biosynthetic_pathways.ipynb`, which also reads
it.

## Running it

```bash
conda activate olfactory-receptors
cd Chemicals
python c00_download_inputs.py      # COCONUT and NCBI taxonomy into data/, about 1.2 GB
```

The scripts are numbered in the order they run.

1. **Build the universe.** `c02_build_universe.py` joins the odorants with COCONUT and computes
descriptors, functional groups and a Joback boiling point for every molecule. Every SMILES is
reduced to its largest fragment and rewritten as an RDKit canonical SMILES, so both sides are
keyed and embedded the same way. The universe has 730,571 molecules.

2. **Embed and map.** `c03_embed.py` embeds every molecule with MoLFormer. It needs a GPU, takes
about an hour on a T1000 and drops the 4,164 molecules whose SMILES are too long for the model
(none of them odorants), leaving 726,407. `c04_umap.py` makes the two-dimensional map from the
full embedding. The map is only for plotting; every distance in the analysis is measured in the
768-dimensional embedding.

3. **Compare odorants with natural products.** `c05_analyze.py` measures how clustered odorants
are, builds a control set of natural products matched to the odorants on boiling point,
molecular weight and logP, and tests pathways and functional groups. `c06_robustness.py` repeats
the matched analysis under five seeds, re-scores it with a scaffold split, and compares the
clustering with matched and random natural-product sets.

4. **Source organisms.** `c07_organism_source.py` resolves COCONUT's organism names against NCBI
taxonomy and compares, taxon by taxon, the share of each molecule's source organisms that falls
in it. `c08_odorant_list_organisms.py` does the kingdom comparison for one list at a time (M2OR
by default, `--panel lg` for Leffingwell and GoodScents). `c12_aquatic_sources.py` labels each
source organism as aquatic or terrestrial.

5. **Screens.** `c09_descriptor_scan.py` tests COCONUT's remaining columns. `c10_source_breadth.py`
tests whether odorants are recorded from more organisms than other natural products; they are
not once database membership is equal, so it is a negative result.

6. **The receptor panel on the map.** `c11_receptors_on_map.py` rebuilds the six co-tuning
clusters and the receptor-class groups from the prediction matrix and places them on the map.
It reads from `Downstream_Analysis/`, `Model_Inputs/` and `Phylogenetic_Analysis/`, and writes
`results/panel_universe_join.csv`, which `c12` and `c13` need.

7. **Figures.** `c13_figures.py` draws every figure as a separate SVG in `figures/`.

`c02` takes about 3 minutes on 64 cores, `c04` about an hour, `c07` about 20 minutes the first
time. The rest take minutes. Only `c03` needs a GPU.

## Outputs

Tables and text reports go to `results/`, figures to `figures/`. The large intermediates
(`universe.parquet`, `coconut_parents.parquet`, `embed_index.csv`, `umap.npy`) are not in Git;
download them with `python scripts/download_zenodo_data.py --chemicals` instead of running `c02`
to `c04`.
