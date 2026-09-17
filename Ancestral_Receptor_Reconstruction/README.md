# Ancestral reconstruction of human olfactory receptors

Reconstructs the sequence of every ancestral olfactory receptor in the human tree, rooted on a
hagfish receptor. The result is 432 ancestral proteins, one per internal node, and they feed the
ancestral analyses in `Downstream_Analysis/notebooks/ancestral/`.

Everything below runs from `human_tree_asr_hagfish_outgroup/`.

## Why two passes

At every column of the alignment there are two questions: did the ancestor have a residue there,
and if so which one. A gap is not a 21st amino acid, and treating it as one lets columns the
ancestor never had drag its sequence around. So the gaps are reconstructed first, on their own.
Then, node by node, the alignment is cut down to the columns that node is inferred to have had,
and only those get amino acids.

## Running it

You need IQ-TREE on your PATH. It is a standalone binary rather than a Python package, so
`environment.yml` does not carry it.

```bash
conda activate olfactory-receptors
cd Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup
```

1. **Root the tree and name the internal nodes.** Run `human_hagfish_tree.ipynb`. It roots the
tree on the hagfish sequence `New|7764.A0A8C4N3S4`, names the internal nodes `node_0` to
`node_432`, and writes `PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames`. Do this
first — every ancestral notebook in the repo reads that file.

2. **Reconstruct the gaps.** `human_hagfish_asr.ipynb` reduces the 434-sequence alignment to a
0/1 presence-absence matrix and reconstructs it on the fixed topology:

```bash
iqtree -m MFP -s PF13853_9606.7764.mafft.presence_absence \
       -te PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames -asr -nt 12
```

Model selection lands on GTR2+FO+R4 over 337 binary sites. `asr_extract.py` reads the `.state`
file into one presence-absence mask per node.

3. **Reconstruct the residues.** For each of the 433 nodes the notebook cuts the alignment down
to that node's kept columns and runs:

```bash
iqtree -s NODE_node_17/human_sequences.node_17.fasta -m LG+F+R9 \
       -te PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames -asr -nt 48
```

`asr_extract.py` then takes the highest-posterior residue at each site. Everything for that node
lands in `NODE_node_17/`.

4. The per-node sequences are concatenated into `all_nodes_ASR.final.fasta`. The copy committed
here is that same file under a clearer name, `hagfish_human_ASR.final.fasta`.

Steps 1 and 2 take seconds. Step 3 is 433 IQ-TREE runs asking for 48 threads each and writes
9.7 GB, so it is a cluster job.

## Using the sequences

`hagfish_human_ASR.final.fasta` holds 432 records named `node_1_ASR` to `node_432_ASR`, ungapped
and 305 to 323 residues long. There is no `node_0_ASR`: node_0 is the root, and IQ-TREE reports
no states for it when the topology is fixed with `-te`.

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

---

These sequences feed [Figure 3](../Downstream_Analysis/notebooks/ancestral/figure3_ancestral_origin.ipynb).
