# Phylogenetic analysis

The trees the downstream analysis stands on: which receptors exist, which class each belongs
to, and how they are related.

```
data/GPCR_classA/     class A GPCRs, to place the ORs inside the wider family
data/HumanTree/       the 433 human ORs -- the tree the ancestral work reconstructs on
data/ReferenceTree/   six species, the tree the model predicts across
notebooks/            tree figures
figures/              rendered trees
```

Three trees, for three jobs.

---

## `HumanTree` — the 433 human ORs

The main tree. Its tips are the extant receptors, and the ancestral reconstruction in
[`../Ancestral_Receptor_Reconstruction/`](../Ancestral_Receptor_Reconstruction/) infers the
internal nodes of exactly this topology.

| file | what it is |
|---|---|
| `human_433_seqs.fa` | the 433 sequences |
| `human_433_seqs.einsi.fa` | MAFFT E-INS-i alignment |
| `human_433_seqs.einsi.gt01.fa` | the same, columns with >10% gaps trimmed |
| `human433_MFP.treefile` | the ML tree (IQ-TREE, ModelFinder chose **LG+F+R10**) |
| `human433_MFP.contree` | the bootstrap consensus tree |
| `human433_MFP_reorder.tree` | tips reordered for plotting — **this is what `reference_tree/02` uses for heatmap row order** |
| `human433_OR_classes.csv` | the class assignment, below |

### `human433_OR_classes.csv`

The file every downstream notebook reads to split receptors by class.

| column | meaning |
|---|---|
| `leaf_id` | tip name in the tree, `9606.<uniprot>` |
| `uniprot` | accession alone |
| `OR_class` | `Class_I` (**62**) or `Class_II` (**371**) |

**The split is exact, not approximate.** The 62 tips under `node_2` of the ASR tree and the
371 under `node_3` are precisely the `Class_I` and `Class_II` sets in this file. That is why
`ancestral_sets.class_clades()` can define the two clades from the tree alone and get the
same answer.

Class I receptors are the smaller, more divergent group — the "fish-like" ORs. They carry a
higher false-positive rate in the model (`reference_tree/03`) and a distinct ligand
chemistry (`reference_tree/02` section 9).

---

## `ReferenceTree` — six species

The tree the model predicts across, and the source of the reference-tree probability matrix
(584 proteins × 754 odorants).

`PF13853.9606_7955_7740_7764_75743_137246.fa` holds the sequences, named `<taxid>.<uniprot>`:

| taxid | species | sequences |
|---|---|---|
| 9606 | human | 433 |
| 7955 | zebrafish | 103 |
| 7764 | hagfish (*Eptatretus burgeri*) | 43 |
| 75743 | lamprey | 2 |
| 137246 | *Callorhinchus milii* | 2 |
| 7740 | *Branchiostoma* | 1 |

**The downstream analysis uses only the 433 human rows.** The other species score far higher
under the human-fitted calibration map — zebrafish averages a 16.9% calibrated binding
probability against human's 2.7% — and none of them are tips of the human ASR tree, so
`ancestral_sets.density_grid()` filters to `9606` before doing anything. If you re-use the
reference matrix elsewhere, apply that filter yourself.

This FASTA is also the sequence source for the exact-sequence join that builds the
experimental layer.

---

## `GPCR_classA` — the wider family

Places the ORs inside class A GPCRs, so "OR class I vs II" can be read against the rest of
the family rather than in isolation. `PF00001.9606.fa` and its clustalo alignment and
FastTree tree, plus three seqid lists: **62** class I, **372** class II, **42**
non-OR class A receptors used as an outgroup.

---

## Notebooks

| notebook | draws |
|---|---|
| `human_tree.ipynb` | the 433-tip tree, coloured |
| `human_tree_c1_c2.ipynb` | the same, class I vs class II |
| `six_species_tree.ipynb` | the reference tree across species |

These are figure-generation only; nothing downstream imports them.

> `six_species_tree.ipynb` has a stored `FileNotFoundError` from a path that no longer
> exists. Fix the path in its first cell before re-running.

---

## Rebuilding

The trees are committed because inference takes hours and the downstream analysis depends on
the exact topology. To rebuild the human tree from the sequences:

```bash
mafft --einsi human_433_seqs.fa > human_433_seqs.einsi.fa
# trim columns with >10% gaps, then:
iqtree -s human_433_seqs.einsi.gt01.fa -m MFP -B 1000 --prefix human433_MFP
```

A rebuilt tree will not have identical branch lengths, and the ancestral reconstruction
would have to be redone against it — so prefer the committed tree unless you specifically
mean to re-infer.

### What is and is not committed

`.gitignore` excludes IQ-TREE working files by pattern (`*.ckp.gz`, `*.iqtree`, `*.state`,
`*.treefile`, `*.contree`, `*.mldist`) because the ancestral reconstruction produces ~11 GB
of them. The **inferred trees and the model-selection reports are explicitly re-included**,
since they are results rather than scratch:

| kept | dropped |
|---|---|
| `human433_MFP.treefile`, `.contree`, `.iqtree` | `.ckp.gz`, `.mldist`, `.log` |
| `ReferenceTree/*.treefile`, `*.iqtree` | everything under `NODE_*/` |

If you add a new tree, check `git status` actually shows it — the broad patterns will
otherwise swallow it silently.
