"""Shared scaffold for the ancestral-repertoire notebooks.

Three reconstructed nodes carry the analysis:

    node_1_ASR -> common_ancestor
    node_2_ASR -> class1_ancestor
    node_3_ASR -> class2_ancestor

Every notebook under notebooks/ancestral/ starts from the same objects — the
thresholded ASR matrix, the ligand presence table, and the activation-pattern
partition — so they are built once here rather than copy-pasted per notebook.

Usage:

    import sys; sys.path.append("../../scripts")
    from ancestral_sets import load_ancestral, FG_SMARTS
    anc = load_ancestral()
    anc.node_presence, anc.pattern_sets, anc.threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

ASR_MATRIX = REPO / "Downstream_Analysis/predictions/aggregated/ASR_Predictions/asr_median_probability_wide.csv"
REF_MATRIX = REPO / "Downstream_Analysis/predictions/aggregated/Reference_Tree_Predictions/reference_median_probability_wide.csv"
PAIRS = REPO / "Model/Data_Preparation/processed/m2or_pairs_model.csv"
FASTA = REPO / "Phylogenetic_Analysis/data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa"
ODORS = REPO / "Downstream_Analysis/predictions/aggregated/ASR_Predictions_RefTree/inchi_smiles_odors.csv"
ASR_TREE = REPO / ("Ancestral_Receptor_Reconstruciton/human_tree_asr_hagfish_outgroup/"
                   "PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames")
CLASSES = REPO / "Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv"

# Where the notebooks in notebooks/ancestral/ write their output.
FIGURES = REPO / "Downstream_Analysis/notebooks/ancestral/Figures"
TABLES = REPO / "Downstream_Analysis/notebooks/ancestral/Tables"

# The ten internal nodes carried through 05. node_1/2/3 are the three of 01-04;
# 4 and 5 sit under the class 1 ancestor, and 6/12/21/32/44 are a nested chain
# descending the class 2 trunk.
TEN_NODES = [1, 2, 3, 4, 5, 6, 12, 21, 32, 44]

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

FG_SMARTS = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "alcohol": "[OX2H][CX4;!$(C=O)]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "ester": "[CX3](=O)[OX2][#6]",
    "ether": "[OD2]([#6])[#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "amide": "[NX3][CX3](=O)[#6]",
    "aromatic_ring": "a1aaaaa1",
    "phenol": "c[OX2H]",
    "nitrile": "[CX2]#N",
    "alkene": "[CX3]=[CX3]",
}

DESCRIPTORS = ["logP", "TPSA", "MW", "HBD", "HBA"]


def save_fig(fig, name: str) -> None:
    """Write a figure to notebooks/ancestral/Figures as SVG.

    SVG only: it is vector, so it stays sharp at any size and diffs as text
    rather than as a new binary blob on every re-run.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.svg", format="svg", bbox_inches="tight")


def save_table(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write a table to notebooks/ancestral/Tables as CSV."""
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=index)


def extant_calibration() -> tuple:
    """Return (threshold, measured_rate) from the reference-tree comparison.

    The threshold is the one 01.model_behaviour derives: the cut at which the
    predicted binding proportion equals the measured one on the 23,782
    experimentally tested cells. The measured rate is that proportion itself.

    Ancestral repertoires are not called at this threshold — see `load_ancestral`
    — but it is reported for reference, and it sets the size of a rank call.
    """
    from Bio import SeqIO

    prob = pd.read_csv(REF_MATRIX, index_col=0)
    human = [i for i in prob.index if str(i).startswith("9606")]
    matrix = prob.loc[human]

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(FASTA, "fasta")}
    seq_to_pid: Dict[str, str] = {}
    for pid in human:
        seq_to_pid.setdefault(sequences[pid], pid)

    pairs = pd.read_csv(PAIRS, usecols=["Species", "Sequence", "Responsive", "smiles_id"])
    human_pairs = pairs[pairs["Species"].str.lower().str.contains("homo")].copy()
    human_pairs["Sequence"] = human_pairs["Sequence"].str.upper().str.strip()
    human_pairs["pid"] = human_pairs["Sequence"].map(seq_to_pid)
    measured = (
        human_pairs[human_pairs["pid"].notna() & human_pairs["smiles_id"].isin(matrix.columns)]
        .groupby(["pid", "smiles_id"])["Responsive"].max()
    )

    observed = measured.values.astype(int)
    predicted = np.array([matrix.at[pid, sid] for pid, sid in measured.index])
    grid = np.arange(0.50, 0.9990, 0.0001)
    rates = np.array([(predicted > t).mean() for t in grid])
    return float(grid[np.abs(rates - observed.mean()).argmin()]), float(observed.mean())


def calibrated_threshold() -> float:
    """The extant threshold alone, for notebooks that only need the number."""
    return extant_calibration()[0]


@dataclass
class Ancestral:
    """Everything the ancestral notebooks share."""

    mode: str                         # "rank" or "absolute"
    top_fraction: float               # rank mode: share of ligands called per node
    per_node_cut: Dict[str, float]    # probability each node's rank cut landed on
    threshold: float                  # the extant calibrated threshold, for reference
    smiles_map: pd.DataFrame          # SML_ID -> SMILES
    predictions: pd.DataFrame         # thresholded ASR matrix
    pred_active: pd.DataFrame         # long format, active pairs only
    node_presence: pd.DataFrame       # SMILES x {common,class1,class2}_ancestor
    pattern_df: pd.DataFrame          # SMILES -> pattern string
    pattern_sets: Dict[str, Set[str]]  # pattern -> SMILES set
    code_sets: Dict[str, Set[str]]     # named groupings used across notebooks
    sets: Dict[str, Set[str]]          # common/class1/class2 + gains/losses

    @property
    def all_smiles(self) -> List[str]:
        return sorted(self.pattern_df["SMILES"].unique())


def load_ancestral(
    mode: str = "absolute",
    top_fraction: Optional[float] = None,
    threshold: float = 0.5,
) -> Ancestral:
    """Build the shared ancestral objects.

    mode="absolute" (default) calls a ligand bound when its median probability
    across the five runs exceeds `threshold`, default 0.5.

    Note this is far more permissive than the 0.915 the extant analysis uses. On
    measured extant pairs a 0.5 cut gives precision 0.27, so roughly three in four
    positives are false. It is used here because the ancestral comparisons are
    internal — node against node — and because reconstructed sequences score
    systematically lower than extant ones, so the extant cut leaves the three
    nodes with 5, 9 and 1 ligands. Absolute repertoire sizes should not be read as
    binding counts.

    mode="rank" is the alternative: each node's top `top_fraction` of ligands,
    defaulting to the measured extant binding proportion. That equalises
    repertoire sizes and removes the reconstruction-confidence gradient.
    """
    extant_thr, measured_rate = extant_calibration()
    thr = float(threshold)
    frac = float(measured_rate if top_fraction is None else top_fraction)

    pairs = pd.read_csv(PAIRS, sep=",")
    smiles_map = (
        pairs[["smiles_id", "SMILES"]].drop_duplicates().rename(columns={"smiles_id": "SML_ID"})
    )

    probabilities = pd.read_csv(ASR_MATRIX)
    ligand_columns = [c for c in probabilities.columns if c.startswith("SML")]
    predictions = probabilities.copy()
    per_node_cut: Dict[str, float] = {}

    if mode == "rank":
        n_call = max(1, int(round(frac * len(ligand_columns))))
        called = probabilities[ligand_columns].copy()
        for row in probabilities.index:
            values = probabilities.loc[row, ligand_columns].astype(float)
            cut = values.nlargest(n_call).iloc[-1]        # inclusive lower bound
            per_node_cut[probabilities.at[row, "protein_id"]] = float(cut)
            called.loc[row] = (values >= cut).astype(int)
        predictions[ligand_columns] = called.astype(int)
    elif mode == "absolute":
        predictions[ligand_columns] = (probabilities[ligand_columns] > thr).astype(int)
    else:
        raise ValueError(f"mode must be 'rank' or 'absolute', got {mode!r}")

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

    node_cuts = {NODE_MAP.get(k, k): v for k, v in per_node_cut.items() if k in NODE_MAP}
    if mode == "absolute":
        node_cuts = {v: thr for v in NODE_MAP.values()}
    return Ancestral(mode, frac, node_cuts, extant_thr, smiles_map, predictions, pred_active,
                     node_presence, pattern_df, pattern_sets, code_sets, sets)


def functional_groups(smiles: List[str]) -> pd.DataFrame:
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


def descriptors(smiles: List[str]) -> pd.DataFrame:
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


def decay_table(threshold: float = 0.5) -> pd.DataFrame:
    """Ligand count against patristic distance from the root, for every node.

    Ancestral nodes come from the ASR matrix, extant receptors from the
    reference-tree matrix; both are placed on the rooted human tree so the two
    can be compared on one axis.
    """
    from ete4 import Tree

    tree = Tree(str(ASR_TREE), parser=1)
    distance = {n.name: tree.get_distance("node_0", n.name)
                for n in tree.traverse() if n.name}

    rows = []
    asr = pd.read_csv(ASR_MATRIX, index_col=0)
    cols = [c for c in asr.columns if c.startswith("SML")]
    for node in asr.index:
        name = node.replace("_ASR", "")
        if name in distance:
            rows.append({"name": name, "kind": "ancestor",
                         "n_ligands": int((asr.loc[node, cols] > threshold).sum()),
                         "distance": distance[name]})

    ref = pd.read_csv(REF_MATRIX, index_col=0)
    cols = [c for c in ref.columns if c.startswith("SML")]
    for tip in ref.index:
        if tip in distance:
            rows.append({"name": tip, "kind": "extant",
                         "n_ligands": int((ref.loc[tip, cols] > threshold).sum()),
                         "distance": distance[tip]})
    return pd.DataFrame(rows)


def _asr_repertoires(threshold: float) -> Dict[str, Set[str]]:
    """Every reconstructed node's repertoire, keyed by tree label, as SMILES."""
    probabilities = pd.read_csv(ASR_MATRIX, index_col=0)
    ligands = [c for c in probabilities.columns if c.startswith("SML")]
    pairs = pd.read_csv(PAIRS, usecols=["smiles_id", "SMILES"]).drop_duplicates("smiles_id")
    to_smiles = dict(zip(pairs["smiles_id"], pairs["SMILES"]))

    binary = probabilities[ligands] > threshold
    out: Dict[str, Set[str]] = {}
    for row in binary.index:
        called = binary.columns[binary.loc[row].values]
        out[str(row).replace("_ASR", "")] = {to_smiles[c] for c in called if c in to_smiles}
    return out


def node_sets(nodes: Optional[List[int]] = None, threshold: float = 0.5) -> Dict[int, Set[str]]:
    """Ligand repertoire of each requested internal node, as a SMILES set."""
    nodes = TEN_NODES if nodes is None else nodes
    repertoires = _asr_repertoires(threshold)
    return {n: repertoires[f"node_{n}"] for n in nodes}


def node_topology(nodes: Optional[List[int]] = None, threshold: float = 0.5) -> pd.DataFrame:
    """Where each node sits on the tree, what it descends to, and what it binds.

    One row per node: its immediate parent, depth from the root, the OR-class
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
    return pd.DataFrame(rows)


def edge_table(threshold: float = 0.5) -> pd.DataFrame:
    """Parent -> child repertoire similarity for every edge of the human tree.

    Ancestral nodes come from the ASR matrix and extant receptors from the
    reference-tree matrix, so an edge may join two ancestors or an ancestor to a
    tip. For each edge:

        similarity  binary Jaccard (Tanimoto) of the two repertoires
        weighted    Jaccard on the raw probabilities, sum(min)/sum(max), which
                    does not depend on the threshold
        retention   share of the parent's ligands the child keeps
        gained/lost ligands the child adds or drops
        branch_len  the child's branch length, i.e. sequence divergence
    """
    from ete4 import Tree

    tree = Tree(str(ASR_TREE), parser=1)

    asr = pd.read_csv(ASR_MATRIX, index_col=0)
    asr.index = [i.replace("_ASR", "") for i in asr.index]
    ref = pd.read_csv(REF_MATRIX, index_col=0)
    ligands = [c for c in asr.columns if c.startswith("SML")]
    prob = pd.concat([asr[ligands], ref[ligands]], axis=0)
    prob = prob[~prob.index.duplicated()]
    binary = (prob > threshold)

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
    return pd.DataFrame(rows)
