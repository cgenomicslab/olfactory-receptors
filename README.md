# Olfactory receptors

Predicting which odorants an olfactory receptor binds — and using those predictions to ask
how odorant recognition evolved.

![Project schema](olfaction_schema.png)

A receptor–ligand model takes a protein sequence and a molecule and returns a binding
probability. Applied across a phylogeny of olfactory receptors it gives something that
cannot be measured directly: the odorant repertoire of receptors that no longer exist.
The analysis runs in two directions — across the 433 extant human receptors, where
predictions can be checked against experiment, and back down the tree to reconstructed
ancestral nodes.

## Layout

```
Chemicals/               odorant datasets, ChEMBL, SMILES to odour-descriptor mapping
Data/                    M2OR receptor-odorant bioassay pairs
Model/
  Data_Preparation/      building the training pairs
  Embeddings/            ESM-C protein and MolFormer molecule embeddings
Phylogenetic_Analysis/   trees: class A GPCRs, six species, the 433 human ORs
Downstream_Analysis/     what the predictions say  -> start here
  notebooks/reference_tree/   extant human receptors
  notebooks/ancestral/        reconstructed ancestral nodes
scripts/                 data download
```

Each analysis folder has its own README. `Downstream_Analysis/README.md` is the one to read
first.

## The model

Proteins are encoded with ESM-C 300M, molecules with MolFormer, and the two are joined by
cross-attention into a binary classifier: does this receptor respond to this odorant?
Training pairs come from M2OR. Every prediction reported here is the **median of five
independent runs**, and the per-run outputs are published so the medians can be
reproduced rather than taken on trust.

## Data

Predictions and embeddings are too large for Git and live in a Zenodo record. Everything
else — sequences, trees, receptor classes, odour descriptors, the analysis code — is in the
repository.

```bash
python scripts/download_zenodo_data.py          # predictions + pooled embeddings
python scripts/download_zenodo_data.py --list   # every asset, with sizes
```

Each archive's SHA-256 is checked before extraction. See `zenodo_manifest.json` and
`Downstream_Analysis/README.md`.

## What the analysis finds

**The threshold is derived, not assumed.** 23,782 receptor–odorant pairs have a measured
M2OR result. 4.7431% of them bind. The threshold used for extant receptors, 0.9150, is the
cut at which the model predicts exactly that proportion. MCC and F1 independently peak at
0.920, so rate-matching lands where the standard metrics already wanted to be. At the 0.5
default the model over-calls binding by 3.6x.

**Class I and class II receptors read different chemistry.** Odorants bound only by class I
are 41% carboxylic acids and more polar; those bound only by class II contain none, and are
more lipophilic. The split is visible in the ancestors too: the class I ancestor's
repertoire is 55% acids while the common ancestor's is 2%, so acid recognition looks like a
class I innovation rather than an inherited trait that class II lost.

**Binding function decays with sequence divergence at a measurable rate.** Across 864
parent–child edges of the human tree, repertoire overlap falls exponentially with branch
length, giving a functional half-life of **0.135 ± 0.021 substitutions per site**. The
result survives removing the threshold entirely.

## Reproducing

The notebooks are numbered and meant to be run in order within each folder. Two thresholds
are in use and the difference matters — 0.9150 for extant receptors, 0.5 for reconstructed
ancestors, for reasons set out in `Downstream_Analysis/README.md`.

An environment specification is not yet included; the analysis runs on Python 3.10 with
pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, RDKit, Biopython and ete4.

## Licence

MIT. See `LICENSE`.

---

Manuscript in preparation.
