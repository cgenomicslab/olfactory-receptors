"""Shared scaffold for the ancestral-repertoire notebooks.

Three reconstructed nodes carry the analysis:

    node_1_ASR -> common_ancestor
    node_2_ASR -> class1_ancestor
    node_3_ASR -> class2_ancestor

Every notebook under ancestral_receptors/ starts from the same objects — the
binarised ASR matrix, the ligand presence table, and the activation-pattern
partition — so they are built once here rather than copy-pasted per notebook.

Binarisation
------------
The default is the density fill from `calibration.py`, the same procedure the
extant notebook uses and driven by the same isotonic map, fitted once on the
23,782 measured human pairs. Applied here with **no experimental overlay**: an
ancestor cannot have measurements, so overlaying them on the extant side only
would put a systematic discontinuity on every ancestor-to-tip edge, exactly
where `edge_table` measures change. Both sides stay prediction-versus-prediction.

The ASR and reference matrices are filled **separately**, each from its own score
distribution. One calibration map, two fills, two cuts — the ASR grid lands near
0.8875 and the extant grid near 0.8947, and both are outputs. Carrying one cut
onto the other matrix would put a chosen number back into the procedure.

Functions that take a `threshold` use the density fill when it is None (the default)
and an absolute probability cut, applied to both matrices alike, when it is a number;
`04.parent_child_decay` sweeps it.

Node names follow the tree file (`node_1` is the common ancestor of the human
receptors). The paper numbers nodes from that common ancestor, so `node_k` is printed
as k - 1; `figure_node` converts.

Usage:

    import sys; sys.path.append("../scripts")
    from ancestral_sets import load_ancestral, FG_SMARTS
    anc = load_ancestral()
    anc.node_presence, anc.pattern_sets, anc.density_cut
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ is not a package; make calibration.py importable however this is loaded
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "scripts"))
import functional_groups as fg  # noqa: E402

ASR_MATRIX = REPO / "Downstream_Analysis/predictions/aggregated/ASR_Predictions/asr_median_probability_wide.csv"
REF_MATRIX = REPO / "Downstream_Analysis/predictions/aggregated/Reference_Tree_Predictions/reference_median_probability_wide.csv"
PAIRS = REPO / "Model_Inputs/Data_Preparation/processed/m2or_pairs_model.csv"
FASTA = REPO / "Phylogenetic_Analysis/data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa"
ODORS = REPO / "Chemicals/processed_data/m2or_inchi_odors_smiles.csv"
ASR_TREE = REPO / ("Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/"
                   "PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames")
CLASSES = REPO / "Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv"

# Where the notebooks in ancestral_receptors/ write their output.
FIGURES = REPO / "Downstream_Analysis/ancestral_receptors/figures"
TABLES = REPO / "Downstream_Analysis/ancestral_receptors/results"

# The ten internal nodes carried through 05. node_1/2/3 are the three of 01-04;
# 4 and 5 sit under the class 1 ancestor, and 6/12/21/32/44 are a nested chain
# descending the class 2 trunk.
TEN_NODES = [1, 2, 3, 4, 5, 6, 12, 21, 32, 44]


def figure_node(node):
    """The number a node carries in the paper's figures and text.

    The tree file names internal nodes from its root down, and its root (node_0) joins
    the hagfish outgroup to the human receptors, so the common ancestor of the human
    repertoire is node_1. The paper counts from that common ancestor instead: node 0 is
    the common ancestor, 1 and 2 the class I and class II ancestors. A repository node_k
    is therefore printed as k - 1. Accepts "node_7", "node_7_ASR" or 7.
    """
    if isinstance(node, str):
        node = int(node.removeprefix("node_").removesuffix("_ASR"))
    return int(node) - 1

# node_1/2/3 in the ASR matrix, in the order used to build activation patterns
NODE_MAP = {
    "node_1_ASR": "common_ancestor",
    "node_2_ASR": "class1_ancestor",
    "node_3_ASR": "class2_ancestor",
}

# Pattern strings read in the row order [class1, common, class2].
# All seven non-empty patterns, so nothing is silently dropped.
PATTERN_ORDER = ["100", "111", "011", "010", "110", "001", "101"]
PATTERN_LABELS = {
    "111": "all three",
    "110": "class1 + common",
    "100": "class1 only",
    "010": "common only",
    "011": "common + class2",
    "001": "class2 only",
    "101": "class1 + class2, not common",
}

# The repository-wide panel (scripts/functional_groups.py), restricted to the groups tracked
# along the ancestral lineages.
ANCESTRAL_GROUPS = ["carboxylic_acid", "alcohol", "aldehyde", "ketone", "ester", "ether",
                    "amine", "amide", "aromatic_ring", "phenol", "nitrile", "alkene"]
FG_SMARTS = {name: fg.FUNCTIONAL_GROUPS[name] for name in ANCESTRAL_GROUPS}

DESCRIPTORS = ["logP", "TPSA", "MW", "HBD", "HBA"]


def save_fig(fig, name: str) -> None:
    """Write a figure to ancestral_receptors/figures/<name>.svg."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", format="svg")


def save_table(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write a table to ancestral_receptors/results as CSV."""
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=index)


_DENSITY_CACHE: dict = {}


def density_grid():
    """Two independent density fills, ASR and extant, sharing one calibration map.

    Returns `(binary, prob, results, ancestral_names)`:

        binary           bool DataFrame, rows x ligands, the calls
        prob             the raw probabilities behind it, same shape
        results          {"asr": FillResult, "extant": FillResult}
        ancestral_names  index labels that came from the ASR matrix

    **The map is shared; the cut is not.** Each matrix is filled from its own
    score distribution, so each gets its own cut as an *output* -- the ASR matrix
    lands near 0.8875, the extant matrix near 0.8947. Filling the two jointly, or
    carrying the extant cut across, would put a chosen number back on the ASR
    grid, which is the thing this procedure exists to avoid. Reconstructions score
    slightly lower than real sequences, and a per-matrix fill absorbs that instead
    of letting it starve the ancestral rows.

    Neither fill takes an experimental overlay. An ancestor cannot have
    measurements, so overlaying them on the extant side alone would put a
    systematic discontinuity on every ancestor-to-tip edge -- exactly where
    `edge_table` measures change.

    Row labels have the `_ASR` suffix stripped so they match tree node names.
    Cached: both fills run once per session and every function here reads them, so
    a node is called identically no matter which entry point asked.

    Extant rows are restricted to human (`9606`). The reference matrix also holds
    zebrafish, hagfish and three other species, and they score far higher under
    the human-fitted map -- zebrafish averages a 16.9% calibrated binding
    probability against human's 2.7%. None of them are tips of the human ASR tree,
    so every function here discards them anyway. The 433 human rows are exactly
    the tree's extant tips.
    """
    if "grid" in _DENSITY_CACHE:
        return _DENSITY_CACHE["grid"]

    from calibration import density_fill, reference_calibration_map

    asr = pd.read_csv(ASR_MATRIX, index_col=0)
    asr.index = [str(i).replace("_ASR", "") for i in asr.index]
    ref = pd.read_csv(REF_MATRIX, index_col=0)
    ref = ref[[str(i).startswith("9606") for i in ref.index]]

    ligands = [c for c in asr.columns if c.startswith("SML") and c in ref.columns]
    asr, ref = asr[ligands], ref[ligands]

    iso = reference_calibration_map(REF_MATRIX, PAIRS, FASTA)
    results = {"asr": density_fill(asr.values, iso, tested_mask=None),
               "extant": density_fill(ref.values, iso, tested_mask=None)}

    binary = pd.concat([
        pd.DataFrame(results["asr"].matrix.astype(bool), index=asr.index, columns=ligands),
        pd.DataFrame(results["extant"].matrix.astype(bool), index=ref.index, columns=ligands),
    ], axis=0)
    prob = pd.concat([asr, ref], axis=0)

    keep = ~binary.index.duplicated()
    binary, prob = binary[keep], prob[keep]

    ancestral = {i for i in asr.index if i in binary.index}
    _DENSITY_CACHE["grid"] = (binary, prob, results, ancestral)
    return _DENSITY_CACHE["grid"]


def _calls(threshold: float | None):
    """Binary calls plus raw probabilities, by density fill or absolute cut.

    `threshold=None` uses the per-matrix density fills; a number is an absolute cut on
    the median probability, applied to both matrices alike.
    """
    binary, prob, _, _ = density_grid()
    if threshold is None:
        return binary, prob
    return (prob > float(threshold)), prob


@dataclass
class Ancestral:
    """Everything the ancestral notebooks share."""

    per_node_cut: dict[str, float]    # probability each node's cut landed on
    smiles_map: pd.DataFrame          # SML_ID -> SMILES
    predictions: pd.DataFrame         # binarised ASR matrix
    pred_active: pd.DataFrame         # long format, active pairs only
    node_presence: pd.DataFrame       # SMILES x {common,class1,class2}_ancestor
    pattern_df: pd.DataFrame          # SMILES -> pattern string
    pattern_sets: dict[str, set[str]]  # pattern -> SMILES set
    code_sets: dict[str, set[str]]     # named groupings used across notebooks
    sets: dict[str, set[str]]          # common/class1/class2 + gains/losses
    density_cut: float                 # the cut that fell out of the ASR fill
    density: float                     # resulting grid density

    @property
    def all_smiles(self) -> list[str]:
        return sorted(self.pattern_df["SMILES"].unique())


def load_ancestral() -> Ancestral:
    """Build the shared ancestral objects for node_1, node_2 and node_3.

    Calls come from the density fill: the isotonic map from `calibration.py`, fitted
    once on the 23,782 measured human pairs, gives every cell a calibrated binding
    probability; their sum is the expected number of true binders; the top that many
    cells by model score are called. No experimental overlay (see the module
    docstring). The cut is an output of the procedure, reported as `density_cut`.
    """
    pairs = pd.read_csv(PAIRS, sep=",")
    smiles_map = (
        pairs[["smiles_id", "SMILES"]].drop_duplicates().rename(columns={"smiles_id": "SML_ID"})
    )

    probabilities = pd.read_csv(ASR_MATRIX)
    ligand_columns = [c for c in probabilities.columns if c.startswith("SML")]
    predictions = probabilities.copy()

    binary, _, results, _ = density_grid()
    asr_fill = results["asr"]          # the ASR matrix's own fill, not the extant one
    stripped = [str(p).replace("_ASR", "") for p in probabilities["protein_id"]]
    calls = binary.reindex(stripped)[ligand_columns].fillna(False).astype(int)
    predictions[ligand_columns] = calls.values
    per_node_cut = {pid: asr_fill.cut for pid in probabilities["protein_id"]}

    subset = predictions[predictions["protein_id"].isin(NODE_MAP)].copy()
    long = subset.melt(
        id_vars="protein_id",
        value_vars=[c for c in subset.columns if c != "protein_id"],
        var_name="SML_ID",
        value_name="active",
    )
    pred_active = long[long["active"] == 1].merge(smiles_map, on="SML_ID", how="left")

    node_presence = (
        pred_active.assign(active=1)
        .pivot_table(index="SMILES", columns="protein_id", values="active",
                     aggfunc="max", fill_value=0)
        .reset_index()
        .rename(columns=NODE_MAP)
    )
    for column in NODE_MAP.values():          # a node with no ligands would be absent
        if column not in node_presence:
            node_presence[column] = 0

    common = set(node_presence.loc[node_presence["common_ancestor"] == 1, "SMILES"])
    class1 = set(node_presence.loc[node_presence["class1_ancestor"] == 1, "SMILES"])
    class2 = set(node_presence.loc[node_presence["class2_ancestor"] == 1, "SMILES"])
    sets = {
        "common": common, "class1": class1, "class2": class2,
        "shared_all": common & class1 & class2,
        "class1_gains": class1 - common, "class2_gains": class2 - common,
        "class1_losses": common - class1, "class2_losses": common - class2,
    }

    # Pattern string is read in row order [class1, common, class2].
    grid = node_presence.set_index("SMILES")[
        ["class1_ancestor", "common_ancestor", "class2_ancestor"]
    ].T
    grid = grid.loc[:, grid.sum(axis=0) > 0]
    patterns = grid.apply(lambda col: "".join(col.astype(int).astype(str).values), axis=0).astype(str)
    pattern_df = pd.DataFrame({"SMILES": grid.columns, "pattern": patterns.loc[grid.columns].values})

    pattern_sets = {
        pat: set(pattern_df.loc[pattern_df["pattern"] == pat, "SMILES"]) for pat in PATTERN_ORDER
    }
    # Sides are symmetric: bound by one class ancestor and not the other.
    code_sets = {
        "class1_side": pattern_sets["100"] | pattern_sets["110"],
        "class2_side": pattern_sets["001"] | pattern_sets["011"],
        "shared": pattern_sets["111"],
        "common_only": pattern_sets["010"],
    }

    node_cuts = {NODE_MAP[k]: v for k, v in per_node_cut.items() if k in NODE_MAP}
    return Ancestral(node_cuts, smiles_map, predictions, pred_active,
                     node_presence, pattern_df, pattern_sets, code_sets, sets,
                     asr_fill.cut, asr_fill.density)


def functional_groups(smiles: list[str]) -> pd.DataFrame:
    """Binary functional-group membership per molecule, one column per group."""
    from rdkit import Chem

    compiled = {name: Chem.MolFromSmarts(s) for name, s in FG_SMARTS.items()}
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        row = {"SMILES": smi}
        for name, patt in compiled.items():
            row[name] = int(mol.HasSubstructMatch(patt)) if mol is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def descriptors(smiles: list[str]) -> pd.DataFrame:
    """RDKit descriptors used across the ancestral notebooks."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors as D, rdMolDescriptors as R

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({"SMILES": smi, **{k: np.nan for k in DESCRIPTORS}})
            continue
        rows.append({"SMILES": smi, "logP": D.MolLogP(mol), "TPSA": D.TPSA(mol),
                     "MW": D.MolWt(mol), "HBD": R.CalcNumHBD(mol), "HBA": R.CalcNumHBA(mol)})
    return pd.DataFrame(rows)


def odor_tags() -> pd.DataFrame:
    """SMILES -> list of odour descriptors, from the in-repo tag table."""
    odors = pd.read_csv(ODORS)
    odors["tags"] = odors["Odors"].fillna("").apply(
        lambda s: [t.strip().lower() for t in str(s).split(",") if t.strip()]
    )
    return odors[["SMILES", "tags"]]


def decay_table(threshold: float | None = None) -> pd.DataFrame:
    """Ligand count against patristic distance from the root, for every node.

    Ancestral nodes come from the ASR matrix, extant receptors from the
    reference-tree matrix; both are placed on the rooted human tree so the two
    can be compared on one axis, and both are binarised the same way.

    `threshold=None` (default) uses the shared density fill; a number is an
    absolute cut. `node_fig` is the paper's number for ancestors, empty for tips.
    """
    from ete4 import Tree

    tree = Tree(str(ASR_TREE), parser=1)
    distance = {n.name: tree.get_distance("node_0", n.name)
                for n in tree.traverse() if n.name}

    binary, _ = _calls(threshold)
    _, _, _, ancestral = density_grid()

    rows = []
    for name in binary.index:
        if name not in distance:
            continue
        rows.append({"name": name,
                     "kind": "ancestor" if name in ancestral else "extant",
                     "n_ligands": int(binary.loc[name].sum()),
                     "distance": distance[name]})
    frame = pd.DataFrame(rows)
    frame.insert(1, "node_fig", _figure_numbers(frame["name"]))
    return frame


def _asr_repertoires(threshold: float | None = None) -> dict[str, set[str]]:
    """Every reconstructed node's repertoire, keyed by tree label, as SMILES."""
    pairs = pd.read_csv(PAIRS, usecols=["smiles_id", "SMILES"]).drop_duplicates("smiles_id")
    to_smiles = dict(zip(pairs["smiles_id"], pairs["SMILES"]))

    binary, _ = _calls(threshold)
    _, _, _, ancestral = density_grid()

    out: dict[str, set[str]] = {}
    for row in binary.index:
        if row not in ancestral:
            continue
        called = binary.columns[binary.loc[row].values]
        out[str(row)] = {to_smiles[c] for c in called if c in to_smiles}
    return out


def node_sets(nodes: list[int] | None = None,
              threshold: float | None = None) -> dict[int, set[str]]:
    """Ligand repertoire of each requested internal node, as a SMILES set."""
    nodes = TEN_NODES if nodes is None else nodes
    repertoires = _asr_repertoires(threshold)
    return {n: repertoires[f"node_{n}"] for n in nodes}


def node_topology(nodes: list[int] | None = None,
                  threshold: float | None = None) -> pd.DataFrame:
    """Where each node sits on the tree, what it descends to, and what it binds.

    One row per node (tree id and the paper's `node_fig`): its immediate parent, depth
    from the root, the OR-class
    composition of its descendant human leaves, its repertoire size, and the
    gains and losses relative to its immediate parent. The parent is taken from
    the full ASR matrix, so it need not itself be one of the requested nodes.
    """
    from ete4 import Tree

    nodes = TEN_NODES if nodes is None else nodes
    tree = Tree(str(ASR_TREE), parser=1)
    index = {n.name: n for n in tree.traverse() if n.name}
    repertoires = _asr_repertoires(threshold)

    classes = pd.read_csv(CLASSES)
    leaf_class = dict(zip(classes["leaf_id"], classes["OR_class"]))

    rows = []
    for n in nodes:
        label = f"node_{n}"
        node = index[label]

        ancestors = []
        cursor = node.up
        while cursor is not None:
            ancestors.append(cursor.name)
            cursor = cursor.up

        leaves = [leaf_class[l.name] for l in node.leaves() if l.name in leaf_class]
        share_i = leaves.count("Class_I") / len(leaves) if leaves else np.nan
        or_class = ("Class_I" if share_i >= 0.9 else
                    "Class_II" if share_i <= 0.1 else "mixed")

        bound = repertoires[label]
        parent = node.up.name if node.up is not None else None
        parent_bound = repertoires.get(parent) if parent else None

        rows.append({
            "node": n,
            "node_fig": figure_node(n),
            "parent": parent,
            "depth": len(ancestors),
            "branch_len": float(node.dist),
            "n_leaves": len(leaves),
            "frac_class_I": round(share_i, 3) if leaves else np.nan,
            "OR_class": or_class,
            "n_ligands": len(bound),
            "n_retained": len(bound & parent_bound) if parent_bound is not None else np.nan,
            "n_gained": len(bound - parent_bound) if parent_bound is not None else np.nan,
            "n_lost": len(parent_bound - bound) if parent_bound is not None else np.nan,
            "ancestors_in_set": ";".join(str(m) for m in nodes if f"node_{m}" in ancestors),
        })
    frame = pd.DataFrame(rows)
    frame.insert(3, "parent_fig", _figure_numbers(frame["parent"]))
    return frame


_TREE_CACHE: dict = {}


def asr_tree():
    """The rooted ASR tree with internal names, parsed once per session.

    `trajectory` asks for a slice at every one of ~865 node times, so re-reading
    the Newick on each call is the difference between seconds and minutes.
    Returns `(tree, index)` where `index` maps node name to node.
    """
    if "tree" not in _TREE_CACHE:
        from ete4 import Tree

        tree = Tree(str(ASR_TREE), parser=1)
        _TREE_CACHE["tree"] = (tree, {n.name: n for n in tree.traverse() if n.name})
    return _TREE_CACHE["tree"]


@dataclass
class Timescale:
    """A relative-time axis over the ingroup, plus the diagnostics behind it."""

    t: dict[str, float]          # node name -> relative time, 0 at ingroup root, 1 at tips
    age: dict[str, float]        # node name -> MPL age in substitutions/site
    root_age: float
    clamped: pd.DataFrame        # nodes whose age exceeded their parent's, and by how much
    kind: str                    # "mpl" or "raw"


def mpl_times(ingroup: str = "node_1") -> Timescale:
    """Ultrametricise the tree by mean path length (Britton et al. 2002).

    Every tip of this tree is a present-day human gene, so the tree *should* be
    ultrametric in time and its non-ultrametricity is rate variation. Making it
    ultrametric is therefore a correction, not an assumption — though see the
    caveat below, which is the reason `raw_times` exists alongside this.

        age(v) = 0                                        if v is a tip
        age(v) = sum_k n_k * (age(k) + bl(k)) / n_v        over children k

    where n_k is k's descendant tip count. The recursion can return a child
    older than its parent, so a root->tips pass clamps `age(v)` to the parent's
    age; `clamped` reports every node that needed it and the violation as a
    fraction of root age. Relative time is then

        t(v) = (age(root) - age(v)) / age(root)

    which is 0 at the ingroup root and 1 at every tip.

    `ingroup` defaults to `node_1`, the gnathostome ancestor. The hagfish tip
    (`node_0`'s other child) only roots the tree and never enters the recursion.

    Caveat carried into every figure that uses this: MPL does not assume a
    strict clock, but it does assume rate variation averages out — the
    assumption most under strain when the rate difference is systematic and
    clade-level, which is exactly the Class I / Class II case here. Plot
    `raw_times` alongside and read the shape, not the timing.
    """
    _, index = asr_tree()
    root = index[ingroup]

    age: dict[str, float] = {}
    n_tips: dict[str, int] = {}
    for node in root.traverse("postorder"):
        if node.is_leaf:
            age[node.name], n_tips[node.name] = 0.0, 1
            continue
        total, count = 0.0, 0
        for child in node.children:
            total += n_tips[child.name] * (age[child.name] + float(child.dist))
            count += n_tips[child.name]
        age[node.name], n_tips[node.name] = total / count, count

    root_age = age[root.name]
    violations = []
    for node in root.traverse("preorder"):
        if node is root or node.is_leaf:
            continue
        if age[node.name] > age[node.up.name]:
            violations.append({
                "node": node.name,
                "parent": node.up.name,
                "excess_frac_root_age": (age[node.name] - age[node.up.name]) / root_age,
                "n_tips": n_tips[node.name],
            })
            age[node.name] = age[node.up.name]

    clamped = (pd.DataFrame(violations, columns=["node", "parent", "excess_frac_root_age", "n_tips"])
               .sort_values("excess_frac_root_age", ascending=False)
               .reset_index(drop=True))
    t = {k: (root_age - v) / root_age for k, v in age.items()}
    return Timescale(t, age, root_age, clamped, "mpl")


def raw_times(ingroup: str = "node_1") -> Timescale:
    """The un-ultrametricised axis: patristic depth from the ingroup root.

    Substitutions per site from `ingroup`, divided by the deepest tip so the
    axis still runs 0 to 1 and can be drawn against `mpl_times`. This assumes
    no clock at all, where MPL assumes rate variation averages out; the truth
    is between them, so the pair brackets the timing rather than pinning it.
    Tips do not land at 1 here — that spread *is* the rate variation.
    """
    _, index = asr_tree()
    root = index[ingroup]

    depth = {root.name: 0.0}
    for node in root.traverse("preorder"):
        if node is root:
            continue
        depth[node.name] = depth[node.up.name] + float(node.dist)

    deepest = max(depth.values())
    t = {k: v / deepest for k, v in depth.items()}
    empty = pd.DataFrame(columns=["node", "parent", "excess_frac_root_age", "n_tips"])
    return Timescale(t, depth, deepest, empty, "raw")


def lineage_slice(times: Timescale, when: float, ingroup: str = "node_1") -> list[str]:
    """The lineages crossing relative time `when` — the repertoire at that moment.

    A branch leading to `v` occupies the interval [t(parent(v)), t(v)), so `v`
    is the lineage crossing `when` exactly on that interval. No ordering of
    receptor addition is needed, which is the point: repertoire *size* is not a
    usable axis because it requires inventing that order.

    Tips are included on the same rule. Under `mpl_times` every tip sits at
    t = 1, so the last slice is the 433 extant receptors; under `raw_times`
    tips are spread and enter at different points.

    Note this returns a *standing* repertoire, not a cumulative one. Crossing a
    node replaces one lineage by its children, so the union of odorants the
    slice detects can fall as `when` increases. That is the estimator, not
    noise — see `trajectory`.
    """
    _, index = asr_tree()
    root = index[ingroup]

    if when <= 0:
        return [root.name]

    out = []
    for node in root.traverse():
        if node is root:
            continue
        lo, hi = times.t[node.up.name], times.t[node.name]
        if lo <= when < hi or (node.is_leaf and when >= hi):
            out.append(node.name)
    return out


def class_clades(ingroup: str = "node_1") -> dict[str, set[str]]:
    """Every node and tip name under the Class I and Class II ancestors.

    Class I is the `node_2` clade and Class II the `node_3` clade. That split is
    not an approximation: the 62 tips under `node_2` and the 371 under `node_3`
    are exactly the Class_I and Class_II sets in `human433_OR_classes.csv`.
    `ingroup` itself belongs to neither, so the class split is undefined at
    t = 0 and every class-stacked panel starts at the node_1 fork.
    """
    _, index = asr_tree()
    return {"Class_I": {n.name for n in index["node_2"].traverse()},
            "Class_II": {n.name for n in index["node_3"].traverse()}}


def trajectory(times: Timescale, grid: np.ndarray | None = None,
               ingroup: str = "node_1", blank_above: int | None = None,
               binary: pd.DataFrame | None = None) -> pd.DataFrame:
    """Sweep the tree and describe the code at every point on the time axis.

    One row per grid point, with the standing repertoire (the lineages crossing
    that time) summarised three ways:

        receptors           n_class_I, n_class_II, n_lineages
        odorants by class   only_I, both, only_II — the class-overlap panel
        detector count      det_1, det_2, det_3, det_4_5, det_6_10, det_11plus
                            plus n_detected and n_single, n_combinatorial

    and, alongside each standing count, the *cumulative* union over everything
    that has existed up to that time (`cum_detected`). The two diverge, and the
    gap is the thing a "code growth" curve hides.

    `blank_above` drops rows binding more than that many odorants before
    summarising — the sensitivity for the hyper-broad reconstructions, two of
    which alone account for a large share of the odorants ever gained.

    `binary` substitutes a different call matrix for the shared density fill,
    indexed by tree node name. The default (None) is the no-overlay fill;
    `overlay_grid()` gives the one whose extant rows carry the measured M2OR values.
    """
    if binary is None:
        binary, _, _, _ = density_grid()
    clades = class_clades(ingroup)
    if grid is None:
        grid = np.round(np.unique(np.clip(list(times.t.values()), 0, 1)), 12)
        grid = np.unique(np.concatenate([[0.0], grid, [1.0]]))

    breadth = binary.sum(axis=1)
    usable = binary.index if blank_above is None else binary.index[breadth <= blank_above]
    bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 9)]
    names = ["det_1", "det_2", "det_3", "det_4_5", "det_6_10", "det_11plus"]

    seen = np.zeros(binary.shape[1], dtype=bool)
    rows = []
    for when in grid:
        members = [m for m in lineage_slice(times, float(when), ingroup)
                   if m in binary.index and m in usable]
        if not members:
            continue
        block = binary.loc[members]
        detectors = block.values.sum(axis=0)
        seen |= detectors > 0

        in_i = [m for m in members if m in clades["Class_I"]]
        in_ii = [m for m in members if m in clades["Class_II"]]
        by_i = binary.loc[in_i].values.sum(axis=0) > 0 if in_i else np.zeros(len(seen), bool)
        by_ii = binary.loc[in_ii].values.sum(axis=0) > 0 if in_ii else np.zeros(len(seen), bool)

        row = {
            "t": float(when),
            "n_lineages": len(members),
            "n_class_I": len(in_i),
            "n_class_II": len(in_ii),
            "n_unassigned": len(members) - len(in_i) - len(in_ii),
            "n_detected": int((detectors > 0).sum()),
            "cum_detected": int(seen.sum()),
            "n_single": int((detectors == 1).sum()),
            "n_combinatorial": int((detectors >= 2).sum()),
            "only_I": int((by_i & ~by_ii).sum()),
            "both": int((by_i & by_ii).sum()),
            "only_II": int((~by_i & by_ii).sum()),
        }
        for name, (lo, hi) in zip(names, bins):
            row[name] = int(((detectors >= lo) & (detectors <= hi)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def edge_table(threshold: float | None = None) -> pd.DataFrame:
    """Parent -> child repertoire similarity for every edge of the human tree.

    Ancestral nodes come from the ASR matrix and extant receptors from the
    reference-tree matrix, so an edge may join two ancestors or an ancestor to a
    tip. For each edge:

        similarity  binary Jaccard (Tanimoto) of the two repertoires
        weighted    Jaccard on the raw probabilities, sum(min)/sum(max), which
                    does not depend on the binarisation at all
        retention   share of the parent's ligands the child keeps
        gained/lost ligands the child adds or drops
        branch_len  the child's branch length, i.e. sequence divergence

    `parent_fig` and `child_fig` are the paper's node numbers (empty for an extant
    child). Edges leaving node_0 have no reconstructed parent and are skipped, as are
    edges whose two repertoires are both empty, where Jaccard is undefined.

    `threshold=None` (default) uses the shared density fill; a number is an absolute
    cut, which is what the sweep in 04 does.
    """
    from ete4 import Tree

    tree = Tree(str(ASR_TREE), parser=1)
    binary, prob = _calls(threshold)

    rows = []
    for node in tree.traverse():
        if node.up is None or not node.name or not node.up.name:
            continue
        child, parent = node.name, node.up.name
        if child not in binary.index or parent not in binary.index:
            continue
        c, p = binary.loc[child].values, binary.loc[parent].values
        union = int((c | p).sum())
        if union == 0:
            continue
        cv, pv = prob.loc[child].values, prob.loc[parent].values
        rows.append({
            "parent": parent, "child": child,
            "kind": "ancestor" if child.startswith("node_") else "extant",
            "similarity": int((c & p).sum()) / union,
            "weighted": float(np.minimum(cv, pv).sum() / np.maximum(cv, pv).sum()),
            "retention": int((c & p).sum()) / int(p.sum()) if p.sum() else np.nan,
            "gained": int((c & ~p).sum()), "lost": int((~c & p).sum()),
            "branch_len": float(node.dist),
        })
    frame = pd.DataFrame(rows)
    frame.insert(1, "parent_fig", _figure_numbers(frame["parent"]))
    frame.insert(3, "child_fig", _figure_numbers(frame["child"]))
    return frame


def _figure_numbers(names) -> pd.Series:
    """Paper node numbers for a column of tree names.

    Empty for extant receptors and for node_0, the root joining the hagfish outgroup,
    which the paper does not number.
    """
    values = [figure_node(n) if str(n).startswith("node_") and str(n) != "node_0" else None
              for n in names]
    return pd.Series(values, dtype="Int64", index=getattr(names, "index", None))


def measured_extant(ref_index, ligands) -> pd.Series:
    """Measured M2OR responses of the human receptors, indexed by (receptor, SML id).

    M2OR rows are joined to receptors by exact amino-acid sequence, not accession:
    M2OR mutants share their parent's accession and must not fold onto it. A pair
    measured more than once counts as responsive if any measurement responded.
    """
    from Bio import SeqIO

    sequences = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(FASTA, "fasta")}
    seq_to_pid: dict[str, str] = {}
    for pid in ref_index:
        seq_to_pid.setdefault(sequences[pid], pid)

    pairs = pd.read_csv(PAIRS, usecols=["Species", "Sequence", "Responsive", "smiles_id"])
    human = pairs[pairs["Species"].str.lower().str.contains("homo")].copy()
    human["Sequence"] = human["Sequence"].str.upper().str.strip()
    human["pid"] = human["Sequence"].map(seq_to_pid)
    keep = human["pid"].notna() & human["smiles_id"].isin(ligands)
    return human[keep].groupby(["pid", "smiles_id"])["Responsive"].max()


def overlay_grid():
    """The density grid with measured M2OR values written over the extant rows.

    Returns `(binary, extant_fill, measured)`. Ancestral rows are the same as in
    `density_grid`, since a reconstructed sequence has no measurements; extant rows
    are filled from their own score distribution with the measured cells held out and
    then overwritten by the measurement.
    """
    from calibration import density_fill, reference_calibration_map

    no_overlay, prob, results, ancestral = density_grid()
    ligands = list(no_overlay.columns)
    extant = [i for i in no_overlay.index if i not in ancestral]
    ref = prob.loc[extant, ligands]

    measured = measured_extant(ref.index, ligands)
    row_of = {p: i for i, p in enumerate(ref.index)}
    col_of = {c: j for j, c in enumerate(ligands)}
    tested = np.zeros(ref.shape, dtype=bool)
    observed = np.zeros(ref.shape, dtype=float)
    for (pid, sid), value in measured.items():
        tested[row_of[pid], col_of[sid]] = True
        observed[row_of[pid], col_of[sid]] = value

    iso = reference_calibration_map(REF_MATRIX, PAIRS, FASTA)
    extant_fill = density_fill(ref.values, iso, tested_mask=tested, experimental=observed)

    binary = no_overlay.copy()
    binary.loc[extant] = extant_fill.matrix.astype(bool)
    return binary, extant_fill, measured
