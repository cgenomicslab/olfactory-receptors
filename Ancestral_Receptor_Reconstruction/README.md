# Ancestral reconstruction

The 432 reconstructed ancestral receptor sequences, one per internal node of the human tree.

## What is here

| File | Is |
|---|---|
| `human_tree_asr_hagfish_outgroup/PF13853_9606.7764.mafft` | the alignment: 433 human receptors plus a hagfish outgroup |
| `...mafft.lg.treefile.rooted.withinternalnames` | the rooted tree with named internal nodes — **every ancestral notebook walks this** |
| `hagfish_human_ASR.final.fasta` | the 432 reconstructed sequences |
| `asr_extract.py`, `tree.py` | the extraction code |
| `human_hagfish_tree.ipynb`, `human_hagfish_asr.ipynb` | how the tree and the reconstruction were run |

## How it was done

Gaps first, then sequence. The alignment is reduced to a presence/absence matrix and
ancestral gap states are reconstructed on the fixed rooted topology (GTR2+FO+R4 over 337
binary sites). For each node the alignment is then cut down to the columns that node is
reconstructed to *have*, and amino acids are reconstructed on the same topology under
LG+F+R9. Each ancestral sequence is the highest-posterior residue at each site.

That gives `node_1` … `node_432`, ungapped length 305–323. IQ-TREE reports no states for
`node_0`, so there are 432 sequences and not 433.

## The per-node runs

The 433 IQ-TREE runs behind these sequences are ~11 GB and are not in Git. You do not need
them — the sequences are here. To audit them:

```bash
python scripts/download_zenodo_data.py --assets asr_node_runs
```

That includes the `.state` files with the full posterior at every site.

What the reconstruction is used for: [Figure
3](../Downstream_Analysis/notebooks/ancestral/figure3_ancestral_origin.ipynb).
