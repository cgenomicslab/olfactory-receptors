# Ancestral reconstruction of human olfactory receptors

Reconstructs the sequence of every ancestral olfactory receptor in the human tree, rooted on a
hagfish receptor (UniProt A0A8C4N3S4). The result is 432 ancestral proteins, one per internal
node, in `human_tree_asr_hagfish_outgroup/hagfish_human_ASR.final.fasta`. They are the input of
`Downstream_Analysis/notebooks/ancestral/`.

## What you need

The environment from the [main README](../README.md), plus Clustal Omega 1.2.3, trimAl 1.5,
MAFFT 7.505 and IQ-TREE 3.0.1, which are not part of it.

Everything below runs from `human_tree_asr_hagfish_outgroup/`.

## How the reconstruction works

At each alignment column there are two questions: did the ancestor have a residue there, and if
so which one. Gaps are reconstructed first, on their own, as presence/absence. Then for each node
the alignment is reduced to the columns that node had, and only those get amino acids. Treating a
gap as a 21st amino acid instead would let columns the ancestor never had change its sequence.

## Steps

1. **Alignment.** The 433 human receptors are aligned and trimmed, and the hagfish receptor is
   added to the trimmed alignment:

   ```bash
   clustalo -i ../../Phylogenetic_Analysis/data/HumanTree/human_433_seqs.fa -o PF13853_9606.clustalo
   trimal -in PF13853_9606.clustalo -out PF13853_9606.clustalo.gt01 -gt 0.1
   mafft --add hagfish.fa PF13853_9606.clustalo.gt01 > PF13853_9606.7764.mafft
   ```

   The result has 434 sequences (433 human and the hagfish receptor) and 337 columns. MAFFT names
   the added sequence `New|7764.A0A8C4N3S4`.

2. **Tree.**

   ```bash
   iqtree -s PF13853_9606.7764.mafft -m MFP -mset LG -B 1000 -T AUTO --prefix PF13853_9606.7764.mafft.lg
   ```

   BIC selects LG+F+R10. Writes `PF13853_9606.7764.mafft.lg.treefile`.

3. **Root and name the nodes.** The first cell of `human_hagfish_asr.ipynb` roots the tree on
   `New|7764.A0A8C4N3S4`, names the internal nodes `node_0` (root) to `node_432`, and writes
   `PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames`. Every ancestral notebook uses
   these node names, so a tree rebuilt from scratch must be renamed and used throughout.

4. **Gaps.** The notebook turns the alignment into a 0/1 presence-absence matrix and runs

   ```bash
   iqtree -m MFP -s PF13853_9606.7764.mafft.presence_absence \
          -te PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames -asr
   ```

   ModelFinder selects GTR2+FO+R4. `asr_extract.py` reads the `.state` file into one 0/1 mask
   per node.

5. **Residues.** For each node the notebook keeps that node's columns and runs

   ```bash
   iqtree -s NODE_node_17/human_sequences.node_17.fasta -m LG+F+R9 \
          -te PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames -asr
   ```

   `asr_extract.py` takes the most probable residue at each site and prints the node's mean
   posterior probability. Each node's files go to `NODE_node_N/`. This step is 433 IQ-TREE runs
   and writes 9.7 GB.

6. The per-node sequences are concatenated into `all_nodes_ASR.final.fasta`, committed as
   `hagfish_human_ASR.final.fasta`.

## Check the output

The notebook does not stop if one IQ-TREE run fails. Afterwards every `NODE_node_*/` except
`node_0` should contain `node_*_ASR.final.fasta`, and the final file should hold 432 records,
`node_1_ASR` to `node_432_ASR`, ungapped, 305 to 323 residues long. There is no `node_0_ASR`:
IQ-TREE reports no states for the root when the topology is fixed with `-te`.

## Using the sequences

A node number is a position in the rooted tree. Three of them carry most of the paper:

* `node_1` — the common ancestor of all 433 human ORs
* `node_2` — the class I ancestor, 62 receptors
* `node_3` — the class II ancestor, 371 receptors

For any other node, read its descendants off the tree:

```python
from ete4 import Tree

t = Tree("PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames", parser=1)
node = next(n for n in t.traverse() if n.name == "node_2")
print(len(list(node.leaves())))   # 62
```

The downstream notebooks go through `Downstream_Analysis/scripts/ancestral_sets.py`, which
caches the parsed tree and uses the same node names.

`human_hagfish_tree.ipynb` opens the tree in an interactive ete4 viewer, with the two classes
coloured and node_1-3 marked.

## The per-node runs

The 433 `NODE_node_*/` directories are 9.7 GB and stay out of Git. You do not need them to use
the sequences. They are on Zenodo if you want to check the reconstruction or redo it under a
different rule, since the `.state` files keep the full posterior at every site rather than just
the residue that won:

```bash
python scripts/download_zenodo_data.py --assets asr_node_runs
```

`prepare_zenodo_asr.py` built that archive: it zips the directories exactly as the pipeline
wrote them and prints the checksum recorded in `zenodo_manifest.json`.
