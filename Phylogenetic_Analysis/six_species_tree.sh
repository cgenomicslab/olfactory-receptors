#!/usr/bin/env bash
# The tree of the 584 olfactory receptors (PF13853) from six species (fig1c).
#
# Input: data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa
#   Homo sapiens 9606 (433), Danio rerio 7955 (103), Eptatretus burgeri 7764 (43),
#   Scyliorhinus torazame 75743 (2), Chiloscyllium punctatum 137246 (2),
#   Branchiostoma lanceolatum 7740 (1)
#
# Run from Phylogenetic_Analysis/:  bash six_species_tree.sh

set -euo pipefail

THREADS=${THREADS:-16}
cd data/ReferenceTree
SIX=PF13853.9606_7955_7740_7764_75743_137246

clustalo -i $SIX.fa -o $SIX.clustalo --threads "$THREADS"
trimal -gt 0.1 -in $SIX.clustalo -out $SIX.clustalo.gt01
FastTree -lg $SIX.clustalo.gt01 > $SIX.clustalo.gt01.lg.fasttree.nw
iqtree -m LG+F+R10 -s $SIX.clustalo.gt01 -te $SIX.clustalo.gt01.lg.fasttree.nw -nt "$THREADS"
