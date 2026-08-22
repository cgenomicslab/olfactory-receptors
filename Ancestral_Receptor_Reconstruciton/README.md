# Ancestral receptor reconstruction

Inferring the amino-acid sequences of the internal nodes of the human OR tree, so the model
can be asked what an ancestral receptor would have bound.

*(The directory name is misspelled — "Reconstruciton". It is referenced by path from
`Downstream_Analysis/scripts/ancestral_sets.py`, so renaming it means updating that
constant too.)*

```
human_tree_asr_hagfish_outgroup/
  PF13853_9606.7764.mafft                          the alignment ASR ran on
  PF13853_9606.7764.mafft.lg.treefile.rooted.
    withinternalnames                              THE tree everything downstream walks
  all_nodes_ASR.final.fasta                        reconstructed sequences, all nodes
  hagfish_human_ASR.final.fasta                    the same, with the outgroup
  hagfish.fa                                       the outgroup sequence
  asr_extract.py                                   pulls one node's sequence from a .state file
  tree.py                                          interactive tree viewer (ete4)
  human_hagfish_tree.ipynb                         tree figure
  NODE_node_*/                                     433 per-node IQ-TREE runs -- NOT in Git
```

## Why a hagfish outgroup

The tree has to be rooted before "ancestral" means anything, and an ingroup cannot root
itself. A hagfish OR sequence is distant enough to sit outside the gnathostome OR radiation
and so defines the root. It plays no other part: `node_0` is the root joining hagfish to
everything else, and **`node_1` — the gnathostome ancestor — is where the analysis actually
starts.** The hagfish tip never enters the ultrametric recursion in
`ancestral_sets.mpl_times`.

## The node naming

Internal nodes are named `node_0` … `node_431` on the rooted tree. In the prediction matrix
they carry an `_ASR` suffix (`node_1_ASR`), which `ancestral_sets.density_grid()` strips so
the labels match tree node names.

| node | what it is |
|---|---|
| `node_0` | the root, joining the hagfish outgroup to the ingroup |
| `node_1` | **the gnathostome OR ancestor** — the common ancestor the analysis starts from |
| `node_2` | ancestor of the 62 Class I receptors |
| `node_3` | ancestor of the 371 Class II receptors |

## What is committed, and what is not

`NODE_node_*/` holds one IQ-TREE ancestral-state run per node — 433 directories, each with a
`.state` file of ~25 MB. **~11 GB in total, and gitignored.** They are the raw posterior
per-site amino-acid distributions; the reconstructed sequences derived from them are
committed instead.

Committed (under 500 KB in total):

| file | why it has to be here |
|---|---|
| `PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames` | **every ancestral notebook walks this.** Without it `ancestral_sets.py` cannot be imported |
| `PF13853_9606.7764.mafft` | the alignment the reconstruction ran on |
| `all_nodes_ASR.final.fasta` | the reconstructed sequences — the model's input |
| `hagfish_human_ASR.final.fasta`, `hagfish.fa` | the same with the outgroup, and the outgroup itself |
| `asr_extract.py`, `tree.py`, `human_hagfish_tree.ipynb` | the code |

`human_hagfish_asr.ipynb` — the notebook that drove the reconstruction — is **not** committed:
it carries 13 MB of IQ-TREE console output. Strip its outputs before adding it if you want
the provenance in Git.

## Reproducing

Reconstruction is the expensive step. Each node's run is an independent IQ-TREE invocation
with `-asr` against the fixed topology, and the sequence is then pulled from the resulting
`.state` file:

```bash
python asr_extract.py human_sequences.node_27.fasta.state node_27
# writes node_27_ASR.fasta, and prints mean/sd/median posterior probability
```

`asr_extract.py` takes a `.state` file and a node name, reconstructs that node's sequence by
taking the maximum-posterior amino acid at each site, and reports the posterior probability
distribution — which is the number to look at when judging how much to trust a node.

Because the `.state` files are not redistributed, **the reconstruction is not reproducible
from this clone**; the reconstructed sequences and the rooted tree are provided as the
outputs instead. Everything downstream of them is fully reproducible.

## Viewing the tree

```bash
python tree.py     # opens an interactive ete4 view, Class I and Class II coloured
```

Requires `ete4`, which `environment.yml` installs.
