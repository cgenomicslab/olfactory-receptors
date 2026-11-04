#!/usr/bin/env bash
# The tree of the 433 human olfactory receptors: the class I / class II assignment
# (notebooks/human_tree_c1_c2.ipynb) and the row order of Extended Data Fig. 3.
#
# Input: data/HumanTree/human_433_seqs.fa, the 433 human sequences of the six-species
# FASTA, written by notebooks/human_tree.ipynb.
#
# Tools: MAFFT 7.505 (E-INS-i), trimAl 1.5, IQ-TREE 3.0.1.
# Run from Phylogenetic_Analysis/:  bash scripts/human_tree.sh

set -euo pipefail

cd data/HumanTree

einsi --thread 4 human_433_seqs.fa > human_433_seqs.einsi.fa
trimal -gt 0.01 -in human_433_seqs.einsi.fa -out human_433_seqs.einsi.gt01.fa

iqtree -s human_433_seqs.einsi.gt01.fa -st AA -mset LG -B 1000 -nt 8 -seed 42 \
       -pre human433_MFP

# human433_MFP_reorder.tree is human433_MFP.treefile reordered with ete4.
