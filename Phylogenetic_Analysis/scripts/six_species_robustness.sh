#!/usr/bin/env bash
# Extended Data Fig. 1c: the six-species tree under four combinations of alignment and
# tree inference. The input is the same 584 sequences as Figure 1c.
#
#   alignment          trimming        tree
#   MAFFT E-INS-i      trimAl -gt 0.1  IQ-TREE
#   MAFFT E-INS-i      trimAl -gt 0.1  FastTree (LG)
#   Clustal Omega      trimAl -gt 0.1  IQ-TREE   (the Figure 1c alignment)
#   MAFFT --auto       trimAl -gt 0.1  IQ-TREE
#
# IQ-TREE: ModelFinder restricted to LG, 1,000 ultrafast bootstrap replicates, fixed seed.
# The three IQ-TREE runs go in parallel, 16 threads each.
#
# Tools: MAFFT 7.505, Clustal Omega 1.2.3, trimAl 1.5, FastTree 2.1.11, IQ-TREE 3.0.1.
# Run from Phylogenetic_Analysis/:  bash scripts/six_species_robustness.sh
# Writes to data/ReferenceTree/robustness/.

set -euo pipefail

THREADS=${THREADS:-16}
SEED=42
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
    iqtree3 -s $SIX.$aln.gt01 --prefix $SIX.$aln.gt01 \
            -mset LG -B 1000 -T "$THREADS" -seed $SEED &
done
wait
