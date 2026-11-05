# Phylogenetic analysis

The receptor trees behind Figure 1 and Extended Data Figs. 1c and 3, and the class I / class II
assignment the rest of the repository uses.

## Trees

| Tree | Sequences | Built by | Drawing | Figure |
|---|---|---|---|---|
| Class A GPCRs | 722 human, Pfam PF00001 | `class_A_tree.sh` | `figures/hGPCR.1.svg` | Fig. 1a |
| Six species | 584 ORs, Pfam PF13853 | `six_species_tree.sh` | `figures/6sp.main.tree.04112026.svg` | Fig. 1c |
| Six species, four methods | the same 584 | `six_species_robustness.sh` | `figures/6sp.unrooted.*.svg` | Ext. Data Fig. 1c |
| Human ORs | 433 | `human_tree.ipynb` | `figures/human433_MFP_reorder.tree.itol.svg` | Ext. Data Fig. 3 |

Run the scripts and notebooks from this folder, e.g. `bash six_species_tree.sh`. They use
HMMER 3.3.2, Clustal Omega 1.2.3, MAFFT 7.505, trimAl 1.5, FastTree 2.1.11 and IQ-TREE 3.0.1
(2.4.0 for the Fig. 1c branch lengths).

`figures/human_hagfish_434.itol.svg` (Fig. 3a) draws the tree built in
[`../Ancestral_Receptor_Reconstruction/`](../Ancestral_Receptor_Reconstruction/).

## Folders

| Folder | Holds |
|---|---|
| `data/GPCR_classA/` | class A sequences and tree, and the tip lists used for the Figure 1 maps |
| `data/ReferenceTree/` | six-species sequences, alignment and tree; `robustness/` holds the four method combinations |
| `data/HumanTree/` | 433 human sequences, alignment and tree, and `human433_OR_classes.csv` |
| `figures/` | tree drawings exported from iTOL |

## Class I and class II

`data/HumanTree/human433_OR_classes.csv` gives the class of each human receptor. The classes
are the two clades at the root of the midpoint-rooted human tree (`human433_MFP_reorder.tree`, the
IQ-TREE tree rooted and reordered in ete4), not taken from gene names.
[`human_tree.ipynb`](human_tree.ipynb) builds the human tree and
[`human_tree_c1_c2.ipynb`](human_tree_c1_c2.ipynb) assigns the classes.

## Drawing the trees

The drawings were made in [iTOL](https://itol.embl.de). To colour the six-species trees, run
`python itol_colours.py` and drop `figures/itol/six_species.itol_colours.txt` onto the
tree in iTOL. Colours follow the species, with human receptors split by class.
