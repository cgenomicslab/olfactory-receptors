#!/usr/bin/env bash
# The tree of the 722 human class A GPCRs (fig1a).
#
# Input: data/GPCR_classA/PF00001.9606.fa, every protein of the human UniProt reference
# proteome carrying Pfam PF00001 (7tm_1, HMM version PF00001.26) above the gathering
# threshold. The proteome is UP000005640_9606.fasta from the UniProt Reference Proteomes,
# one protein per gene, 20,654 sequences. The search returns exactly these 722 proteins:
#
#   hmmsearch -o /dev/null --tblout PF00001.9606.tblout --cut_ga PF00001.hmm UP000005640_9606.fasta
#
# The hits were written to PF00001.9606.fa with headers renamed to 9606.<UniProt accession>.
#
# Tools: HMMER 3.3.2, Clustal Omega 1.2.3, trimAl 1.5, FastTree 2.1.11.
# Run from Phylogenetic_Analysis/:  bash scripts/class_A_tree.sh

set -euo pipefail

THREADS=${THREADS:-8}
cd data/GPCR_classA

clustalo -i PF00001.9606.fa -o PF00001.9606.clustalo --threads "$THREADS"
trimal -gt 0.1 -in PF00001.9606.clustalo -out PF00001.9606.clustalo.gt01
FastTree -lg PF00001.9606.clustalo.gt01 > PF00001.9606.clustalo.gt01.lg.fasttree
