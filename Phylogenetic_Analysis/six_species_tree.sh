#!/usr/bin/env bash
# The tree of the 584 olfactory receptors (PF13853) from six species (fig1c).
#
# Input: data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa
#   Homo sapiens 9606 (433), Danio rerio 7955 (103), Eptatretus burgeri 7764 (43),
#   Scyliorhinus torazame 75743 (2), Chiloscyllium punctatum 137246 (2),
#   Branchiostoma lanceolatum 7740 (1)
#
#   alignment        MAFFT E-INS-i
#   trimming         trimAl -gt 0.1
#   model            ModelFinder best-fit model by BIC (JTT+F+R10)
#   tree             IQ-TREE, three searches (seeds 42, 1, 2); the search with the
#                    highest likelihood is kept (seed 1)
#   branch support   SH-aLRT (1,000 replicates) and ultrafast bootstrap (1,000 replicates)
#
# Output: $SIX.einsi.gt01.ML.best.treefile        the tree
#         $SIX.einsi.gt01.ML.best.alrt.treefile   the tree with SH-aLRT supports
#         $SIX.einsi.gt01.ML.best.ufb.suptree     the tree with ultrafast bootstrap supports
#
# Run from Phylogenetic_Analysis/:  bash six_species_tree.sh

set -euo pipefail

THREADS=${THREADS:-16}
cd data/ReferenceTree
SIX=PF13853.9606_7955_7740_7764_75743_137246
ALN=$SIX.einsi.gt01

# MAFFT's iterative refinement is not deterministic with more than one thread, so a rerun
# places some gaps differently. The tree was built from the $SIX.einsi stored here.
einsi --thread "$THREADS" $SIX.fa > $SIX.einsi
trimal -gt 0.1 -in $SIX.einsi -out $ALN

# Model selection. The best-fit model is in $ALN.MFall.iqtree.
iqtree3 -s $ALN -m MF -mset LG,WAG,JTT,JTTDCMut,VT,PMB,Blosum62,Dayhoff,DCMut,Q.pfam \
        -cmax 15 -T 4 --prefix $ALN.MFall
MODEL=JTT+F+R10

# Three searches. Seed 42 also runs the ultrafast bootstrap and saves the bootstrap trees.
iqtree3 -s $ALN -m $MODEL -B 1000 -bnni -wbt -alrt 1000 -seed 42 -T 8 --prefix $ALN.ML.s42 &
iqtree3 -s $ALN -m $MODEL -seed 1 -T 8 --prefix $ALN.ML.s1 &
iqtree3 -s $ALN -m $MODEL -seed 2 -T 8 --prefix $ALN.ML.s2 &
wait

# Keep the search with the highest log-likelihood.
BEST=42
BEST_LNL=$(grep -m1 'BEST SCORE FOUND' $ALN.ML.s42.log | awk '{print $NF}')
for seed in 1 2; do
    lnl=$(grep -m1 'BEST SCORE FOUND' $ALN.ML.s$seed.log | awk '{print $NF}')
    if awk -v a="$lnl" -v b="$BEST_LNL" 'BEGIN {exit !(a > b)}'; then
        BEST=$seed
        BEST_LNL=$lnl
    fi
done
echo "best search: seed $BEST, lnL $BEST_LNL"
cp $ALN.ML.s$BEST.treefile $ALN.ML.best.treefile

# Supports on the kept tree: ultrafast bootstrap from the seed-42 bootstrap trees,
# and SH-aLRT with the topology fixed.
iqtree3 -sup $ALN.ML.best.treefile -t $ALN.ML.s42.ufboot --prefix $ALN.ML.best.ufb
iqtree3 -s $ALN -m $MODEL -te $ALN.ML.best.treefile -alrt 1000 -T 8 --prefix $ALN.ML.best.alrt
