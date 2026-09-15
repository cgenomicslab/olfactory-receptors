# Phylogenetics

The trees, and the class I / class II assignment everything else depends on.

Start with
[`notebooks/figure1_receptor_phylogeny.ipynb`](notebooks/figure1_receptor_phylogeny.ipynb).
It builds Figure 1 and Extended Data Figure 1, shows the panels inline, and lists the exact
commands used to build each tree.

## Three trees

| Tree | Sequences | Used for |
|---|---|---|
| Class A GPCRs | 722 human, Pfam PF00001 | Figure 1a,b — are ORs one clade? |
| Six species | 584, Pfam PF13853 | Figure 1c,d — when did the classes split? |
| Human ORs | 433 | class assignment; the row order in Extended Data Fig. 3 |

## Folders

| Folder | Holds |
|---|---|
| `data/GPCR_classA/` | the class A alignment, tree and tip lists |
| `data/ReferenceTree/` | the six-species alignments and trees, including the four method combinations of Extended Data Fig. 1c |
| `data/HumanTree/` | the 433-receptor alignment and tree, and `human433_OR_classes.csv` |
| `figures/` | the tree drawings, exported from iTOL |

## The one file everything reads

`data/HumanTree/human433_OR_classes.csv` says which class each of the 433 receptors belongs
to. It is not from gene names — it is the two clades descending from the root of the human
tree. The Figure 1 notebook rebuilds it and checks it matches.

Tree drawings come from [iTOL](https://itol.embl.de), a web service, so they are committed
as SVG rather than rebuilt by a script. The trees they draw are all here.
