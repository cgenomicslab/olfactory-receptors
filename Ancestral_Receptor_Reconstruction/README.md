# Ancestral receptor reconstruction

Reconstructs the amino-acid sequences of the 432 internal nodes of the human OR tree, so
the binding model can be asked what an ancestral receptor would have bound.

Everything happens in `human_tree_asr_hagfish_outgroup/human_hagfish_asr.ipynb`.

## The output that matters

`hagfish_human_ASR.final.fasta` — 432 ancestral sequences, `node_1` to `node_432`.
These are the rows of the ASR prediction matrix.

`PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames` — the rooted tree with
internal node names. Every notebook in `Downstream_Analysis/notebooks/ancestral/` walks it.

## The steps

1. Align 433 human ORs plus one hagfish sequence (MAFFT).
2. Build the tree with IQ-TREE, root it on the hagfish, name the internal nodes
   `node_0` … `node_432`.
3. Recode the alignment as presence/absence (gap = 0, residue = 1) and run IQ-TREE `-asr`
   on the fixed tree. This says which alignment columns each ancestor had.
4. For each node: keep only its present columns, run IQ-TREE `-asr` again, and take the
   highest-posterior amino acid at each site.

The hagfish roots the tree. That is what makes `node_1` the common ancestor of Class I
(`node_2`) and Class II (`node_3`). `node_0` is the root itself and has no reconstruction —
IQ-TREE does not report states for it — so the usable set starts at `node_1`.

## Files

| | |
|---|---|
| `PF13853_9606.7764.mafft` | the alignment, 434 sequences |
| `hagfish.fa` | the outgroup sequence |
| `asr_extract.py` | pulls one node's sequence out of an IQ-TREE `.state` file |
| `tree.py` | opens the tree in an interactive viewer |
| `human_hagfish_tree.ipynb` | the tree figure |

`NODE_node_*/` holds the 433 per-node IQ-TREE runs — 9.6 GB, too large for Git. They are
archived on Zenodo instead:

```bash
python scripts/download_zenodo_data.py --assets asr_node_runs
```

That unpacks the `NODE_node_*` tree back into `human_tree_asr_hagfish_outgroup/`. You only
need it to audit the reconstruction — the `.state` files hold the per-site posterior over
all 20 amino acids, so with them you can check confidence at any site or re-extract under a
different rule. The finished sequences are in the repository, so nothing downstream needs
the archive.

`prepare_zenodo_asr.py` builds it.

## Re-running one node

```bash
python asr_extract.py NODE_node_27/human_sequences.node_27.fasta.state node_27
```

Writes `node_27_ASR.fasta` and prints the mean posterior probability, which is the number
to look at when judging how much to trust a node.
