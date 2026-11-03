# Chemicals

Compares every known odorant with the natural products in COCONUT: where odorants sit on a
map of chemical space, which biosynthetic pathways and functional groups they carry, and
which organisms they come from. The plots of Figure 4 and Extended Data Fig. 7 are drawn
here.

## What you need

The environment from the [main README](../README.md). `c03` needs an NVIDIA GPU.

To redraw the figures without running `c02`–`c04`, download the intermediates instead:

```bash
python scripts/download_zenodo_data.py --chemicals             # universe, embedding index, map, 175 MiB
python scripts/download_zenodo_data.py --chemical-embeddings   # the 768-d embedding for c05 and c06, 2 GB
```

## The odorants and the background

| set | molecules | used for |
|---|---|---|
| M2OR ligands | 754 | the receptor predictions and all of `Downstream_Analysis/` |
| all odorants | 5,962 | this folder |

The 5,962 are M2OR (754), Leffingwell (3,522) and GoodScents (4,492), merged on InChIKey.
They are in `processed_data/`:

| file | contents | written by |
|---|---|---|
| `m2or_inchi_odors_smiles.csv` | the 754 M2OR ligands and their odour descriptors | `c01_odorant_sets.ipynb` |
| `odorants_m2or_leffingwell_goodscents.csv` | the 5,962 odorants and the lists each is in | `c01_odorant_sets.ipynb` |
| `ligand_pathways.csv` | the NPClassifier pathway of each M2OR ligand, from COCONUT | `build_ligand_pathways()` in `Downstream_Analysis/scripts/activation_code.py` |

The background is COCONUT release 08-2026, 738,827 entries. Every SMILES is reduced to its
largest fragment, written as an RDKit canonical SMILES and keyed by InChIKey. Entries with
the same key are merged, which leaves 727,582 structures. The 2,973 odorants found in
COCONUT are removed from the background, so no molecule is on both sides. MoLFormer cannot
read 4,164 SMILES longer than 202 tokens, none of them odorants. The analysis therefore has
726,407 molecules: 5,962 odorants and 720,445 natural products.

## Steps

Everything runs from `Chemicals/`, in the order of the numbers.

1. **Inputs.** Downloads COCONUT 08-2026 and the NCBI taxonomy of 2026-08-01 into `data/`,
   checks both against their SHA-256 and builds `data/taxa.sqlite` for ete4.

   ```bash
   python c00_download_inputs.py
   ```

2. **Odorant sets.** Reads M2OR from `../Data/m2or_pairs.csv` and Leffingwell and
   GoodScents from the pyrfume-data repository at a pinned commit, and writes the two odorant
   files above.

   ```bash
   jupyter nbconvert --to notebook --execute --inplace c01_odorant_sets.ipynb
   ```

3. **Universe.** Joins the odorants with COCONUT and computes for every molecule the RDKit
   descriptors, the functional groups of `scripts/functional_groups.py` and a Joback boiling
   point. Writes `results/universe.parquet` (730,571 molecules).

   ```bash
   python c02_build_universe.py
   ```

4. **Embedding.** MoLFormer-XL (`ibm-research/MoLFormer-XL-both-10pct`, pinned revision),
   768 dimensions per molecule. Writes `results/embeddings.npy` and `results/embed_index.csv`
   (726,407 rows). GPU.

   ```bash
   python c03_embed.py
   ```

5. **Map.** UMAP of the 768-d embedding (cosine, 15 neighbours, min_dist 0.1, seed 42).
   The map is only for plotting; every distance in the analysis is measured in the 768-d
   embedding. Writes `results/umap.npy`.

   ```bash
   python c04_umap.py
   ```

6. **Odorants against natural products.**
   * clustering: the share of each odorant's 50 nearest neighbours that are odorants
   * a matched control: each odorant paired with the closest unused natural product in
     boiling point, molecular weight and logP (caliper 0.25 SD)
   * biosynthetic pathways: Fisher's exact test against all natural products
   * functional groups: odds ratio against the matched partners, exact McNemar test, interval
     from 2,000 resamples of the pairs

   Benjamini–Hochberg q-values throughout.

   ```bash
   python c05_analyze.py
   ```

7. **Robustness.** Repeats the matching under five seeds, re-scores the separability with a
   scaffold split, and compares the clustering with matched and random natural-product sets.
   Writes the three robustness tables of the Supplementary Tables.

   ```bash
   python c06_robustness.py
   ```

8. **Source organisms.** Resolves COCONUT's organism names to NCBI taxa with ete4 and, for
   every taxon from domain to genus, compares the mean share of a molecule's source organisms
   that falls in it: permutation test (20,000 shuffles), bootstrap 95% interval on the
   ratio, q-values within each rank. `c08` repeats the kingdom comparison for one odorant
   list at a time.

   ```bash
   python c07_organism_source.py
   python c08_odorant_list_organisms.py              # M2OR
   python c08_odorant_list_organisms.py --panel lg   # Leffingwell and GoodScents
   ```

9. **Receptors on the map.** Places the receptor classes and co-tuning clusters of
   `Downstream_Analysis/` on the map (needs the predictions, see the main README), and labels
   each source organism aquatic or terrestrial for `Supplementary_Material/aquatic_origin/`.

   ```bash
   python c09_receptors_on_map.py
   python c10_aquatic_sources.py
   ```

10. **Figures.** Draws every plot as its own SVG in `figures/`.

    ```bash
    python c11_figures.py
    ```

## Which file is which figure

| figure | file | drawn from |
|---|---|---|
| Fig. 4a | `odorant_map.svg` | `umap.npy`, `universe.parquet` |
| Fig. 4b | `organism_kingdom.svg` | `organism_rank_enrichment.csv` |
| Fig. 4c | `biosynthetic_pathway.svg` | `pathway_enrichment.csv` |
| Fig. 4d | `functional_groups_matched.svg` | `fg_enrichment.csv` |
| Ext. Data Fig. 7a | `aromatic_map.svg` | `umap.npy`, `universe.parquet` |
| Ext. Data Fig. 7b | `aromatic_vs_aliphatic.svg` | `universe.parquet` |
| Ext. Data Fig. 7c–e | `receptor_class_map_class_I.svg`, `_shared.svg`, `_class_II.svg` | `panel_universe_join.csv` |
| Ext. Data Fig. 7f | `receptor_class_volatility.svg` | `panel_universe_join.csv`, `universe.parquet` |
| Ext. Data Fig. 7g | `organism_order.svg` | `organism_rank_enrichment.csv` |
| Ext. Data Fig. 7h | `organism_family.svg` | `organism_rank_enrichment.csv` |

## Outputs

Tables and text reports go to `results/`, plots to `figures/`. The large intermediates
(`universe.parquet`, `coconut_parents.parquet`, `embed_index.csv`, `embeddings.npy`,
`umap.npy`, `aquatic_sources.csv`, `aquatic_taxa_seen.csv`) are on Zenodo.
