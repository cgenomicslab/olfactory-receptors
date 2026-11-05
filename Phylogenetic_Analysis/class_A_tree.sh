#!/usr/bin/env bash
# The tree of the 722 human class A GPCRs (fig1a).
#
# Input: data/GPCR_classA/PF00001.9606.fa, the 722 class A GPCRs of the 
# human reference proteome (Pfam PF00001.26).
# Generated via hmmsearch from UP000005640_9606.fasta:
#   hmmsearch --cut_ga --tblout PF00001.9606.tblout PF00001.hmm UP000005640_9606.fasta
#
# Headers in PF00001.9606.fa are renamed to 9606.<UniProt accession>.
#
# Run from Phylogenetic_Analysis/:  bash class_A_tree.sh

set -euo pipefail

THREADS=${THREADS:-8}
cd data/GPCR_classA

clustalo -i PF00001.9606.fa -o PF00001.9606.clustalo --threads "$THREADS"
trimal -gt 0.1 -in PF00001.9606.clustalo -out PF00001.9606.clustalo.gt01
FastTree -lg PF00001.9606.clustalo.gt01 > PF00001.9606.clustalo.gt01.lg.fasttree
