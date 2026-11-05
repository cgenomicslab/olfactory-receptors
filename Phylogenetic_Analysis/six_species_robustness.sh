#!/usr/bin/env bash
# The six-species tree under four combinations of alignment and
# tree inference. The input is the same 584 sequences as Figure 1c (Ext. Data fig1c).
#
#   alignment          trimming        tree
#   MAFFT E-INS-i      trimAl -gt 0.1  IQ-TREE
#   MAFFT E-INS-i      trimAl -gt 0.1  FastTree (LG)
#   Clustal Omega      trimAl -gt 0.1  IQ-TREE  (the Figure 1c alignment)
#   MAFFT --auto       trimAl -gt 0.1  IQ-TREE
#
# IQ-TREE: ModelFinder restricted to LG, 1,000 ultrafast bootstrap replicates, fixed seed.
# The three IQ-TREE runs go in parallel, 16 threads each.
#
# Run from Phylogenetic_Analysis/:  bash six_species_robustness.sh
# Writes to data/ReferenceTree/robustness/.

set -euo pipefail

THREADS=${THREADS:-16}
SIX=PF13853.9606_7955_7740_7764_75743_137246

mkdir -p data/ReferenceTree/robustness
cd data/ReferenceTree/robustness

einsi --thread "$THREADS" ../$SIX.fa > $SIX.einsi
mafft --auto --thread "$THREADS" ../$SIX.fa > $SIX.mft
cp ../$SIX.clustalo.gt01 $SIX.clustalo.gt01

trimal -gt 0.1 -in $SIX.einsi -out $SIX.einsi.gt01
trimal -gt 0.1 -in $SIX.mft   -out $SIX.mft.gt01

FastTree -lg $SIX.einsi.gt01 > $SIX.einsi.gt01.lg.fasttree

for aln in einsi clustalo mft; do
    iqtree -s $SIX.$aln.gt01 --prefix $SIX.$aln.gt01 \
            -mset LG -B 1000 -T "$THREADS" -seed 42 &
done
wait
