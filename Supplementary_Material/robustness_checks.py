"""Analysis helpers shared by the robustness notebooks in Supplementary_Material/.

The notebooks `cluster_number`, `calling_rule` and `model_runs` ask the same question from three
sides: do the paper's results depend on one choice the analysis had to make? To answer it they
rebuild the binding calls in another way and recompute the same short list of results. This module
holds the code for both steps, so every notebook recomputes a result in exactly the same way.

The results recomputed are the ones the text rests on:

- extant receptors: tuning breadth by class, the chemistry of the odorants each class reads, and
  the six activation clusters with their odour descriptors;
- ancestral receptors: the repertoires of the first nodes, the chemistry gained along the two
  founding duplications, the combinatorial-coding trajectory and the decay of parent-child
  similarity with branch length.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy.optimize import curve_fit
from scipy.stats import fisher_exact, rankdata
from sklearn.metrics import adjusted_rand_score

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "Downstream_Analysis/scripts"))
sys.path.append(str(REPO / "scripts"))

import activation_code as ac  # noqa: E402
import ancestral_sets as an  # noqa: E402
import functional_groups as fg  # noqa: E402
from calibration import reference_calibration_map  # noqa: E402

PROBABILITIES = REPO / "Downstream_Analysis/predictions/aggregated/Reference_Tree_Predictions/reference_median_probability_wide.csv"
M2OR_PAIRS = REPO / "Model_Inputs/Data_Preparation/processed/m2or_pairs_model.csv"
ODOURS = REPO / "Chemicals/processed_data/m2or_inchi_odors_smiles.csv"
FASTA = REPO / "Phylogenetic_Analysis/data/ReferenceTree/PF13853.9606_7955_7740_7764_75743_137246.fa"
CLASSES = REPO / "Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv"
MAIN_CLUSTERS = REPO / "Downstream_Analysis/extant_receptors/results/odorant_clusters.csv"
# Per-run predictions: where the Zenodo download unpacks them (`runs/`), or, on the machine that
# trained the model, the original archive the Zenodo release was made from.
_ZENODO_RUNS = REPO / "Downstream_Analysis/predictions/runs"
_ARCHIVE_RUNS = (REPO / "Downstream_Analysis/predictions/archive/Predictions_M2OR-20251005T081853Z-1-001"
                 / "Predictions_M2OR/All_Data_No_Mixtures_Predictions")
RUNS = _ZENODO_RUNS if _ZENODO_RUNS.is_dir() else _ARCHIVE_RUNS
RUN_METRICS = (REPO / "Downstream_Analysis/predictions/model_run_evaluation_metrics.csv"
               if _ZENODO_RUNS.is_dir() else _ARCHIVE_RUNS / "Runs_Model_Evaluation_Metrics.csv")

# Text sizes of the supplementary figures, the same as the paper's Fig. 3 panels: every figure is
# drawn at 288 x 216 pt and placed in Inkscape without scaling.
TEXT_SIZES = {"axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
              "legend.fontsize": 7}

# Descriptors that mark the acid/dairy cluster of the paper (Table 1, cluster I).
ACID_CLUSTER_DESCRIPTORS = {"acidic", "sour", "cheesy"}

# The two founding duplications, as tree labels: paper node 0 -> 1 and paper node 2 -> 5.
FOUNDING_DUPLICATIONS = {
    "duplication 1 (node 0 -> 1), carboxylic acid": ("node_1", "node_2", "carboxylic_acid"),
    "duplication 2 (node 2 -> 5), aromatic ring": ("node_3", "node_6", "aromatic_ring"),
}


# ---------------------------------------------------------------------------------------------
# Calling rules
# ---------------------------------------------------------------------------------------------

def rank_fill(scores, calibration_map, factor=1.0, tested=None):
    """Call the top `factor` x N untested cells by score, N being the calibrated number of binders.

    With `factor=1` this is the density fill of `calibration.py`: N is the sum of the calibrated
    binding probabilities of the untested cells. Other factors call half, or twice, as many
    cells, which is how the notebooks ask whether the calibrated number of calls matters.

    Parameters
    ----------
    scores : ndarray
        Model probabilities, receptors x odorants.
    calibration_map : IsotonicRegression
        Map from model score to measured binding frequency.
    factor : float
        Multiple of N to call.
    tested : ndarray of bool, optional
        True where a measurement exists. Those cells are left out of the ranking, as in the
        density fill; the measured values are written over them by the caller.

    Returns
    -------
    ndarray of int8
        Binary calls, same shape as `scores`.
    """
    tested = np.zeros(scores.shape, bool) if tested is None else tested
    untested = ~tested
    untested_scores = scores[untested]
    n_expected = int(round(float(calibration_map.predict(untested_scores).sum())))
    n_called = int(round(factor * n_expected))

    calls = np.zeros(scores.shape, np.int8)
    untested_positions = np.flatnonzero(untested.ravel())
    highest_first = np.argsort(untested_scores)[::-1]
    calls.ravel()[untested_positions[highest_first[:n_called]]] = 1
    return calls


def cut_fill(scores, cut):
    """Call every cell whose model probability is above a fixed cut."""
    return (scores > cut).astype(np.int8)


def overlay(calls, data):
    """Write the measured values over the calls, as the hybrid matrix does."""
    return np.where(data.tested, data.experimental, calls).astype(int)


# ---------------------------------------------------------------------------------------------
# Extant receptors
# ---------------------------------------------------------------------------------------------

@dataclass
class ExtantInputs:
    """What the extant summary needs besides the call matrix."""

    data: object                 # ActivationData of activation_code
    odorant_ids: np.ndarray
    tags_by_odorant: dict        # odorant -> odour descriptors
    smiles_by_odorant: dict      # odorant -> SMILES
    class_of: dict               # receptor -> "I" or "II"
    is_acid: np.ndarray          # one entry per odorant
    is_aromatic: np.ndarray
    main_cluster_of: dict        # odorant -> cluster in the paper


def has_group(smiles_by_odorant, odorant_ids, group):
    """Whether each odorant carries a functional group of `scripts/functional_groups.py`."""
    pattern = Chem.MolFromSmarts(fg.FUNCTIONAL_GROUPS[group])
    return np.array([Chem.MolFromSmiles(smiles_by_odorant[odorant]).HasSubstructMatch(pattern)
                     for odorant in odorant_ids])


def load_extant_inputs():
    """Load the median predictions, the measurements, descriptors, classes and paper clusters."""
    data = ac.load_activation_data(PROBABILITIES, M2OR_PAIRS, FASTA)
    smiles_by_odorant, tags_by_odorant = ac.odour_tags(M2OR_PAIRS, ODOURS, data.odorant_ids)
    main_clusters = pd.read_csv(MAIN_CLUSTERS)
    return ExtantInputs(
        data=data, odorant_ids=np.array(data.odorant_ids), tags_by_odorant=tags_by_odorant,
        smiles_by_odorant=smiles_by_odorant,
        class_of=ac.receptor_classes(CLASSES),
        is_acid=has_group(smiles_by_odorant, data.odorant_ids, "carboxylic_acid"),
        is_aromatic=has_group(smiles_by_odorant, data.odorant_ids, "aromatic_ring"),
        main_cluster_of=dict(zip(main_clusters.smiles_id, main_clusters.cluster)))


def cluster_perception(odorant_ids, labels, tags_by_odorant, is_acid, k=ac.N_CLUSTERS):
    """How perceptually coherent a partition of the odorants is.

    Parameters
    ----------
    odorant_ids : ndarray of str
        The clustered odorants.
    labels : ndarray of int
        Cluster of each odorant.
    tags_by_odorant : dict
        Odour descriptors of each odorant.
    is_acid : ndarray of bool
        Carboxylic acid flag, aligned to `odorant_ids`.
    k : int
        Number of clusters.

    Returns
    -------
    dict
        clusters_with_descriptor: clusters with at least one significantly enriched descriptor
        (hypergeometric, Benjamini-Hochberg q < 0.05, as in Table 1); significant_enrichments;
        themes: the odour themes of `activation_code.THEME_RULES` found among the enrichments;
        acid_cluster_pct_acid: share of carboxylic acids in the most acid-rich cluster;
        acid_cluster_is_sour: whether that cluster carries acidic, sour or cheesy.
    """
    enrichment = ac.tag_enrichment(odorant_ids, labels, tags_by_odorant, k=k)
    significant = ac.significant_enrichment(enrichment)

    themes = set()
    for theme_descriptors, theme_name, _ in ac.THEME_RULES:
        if significant.tag.isin(theme_descriptors).any():
            themes.add(theme_name)

    acid_share = [is_acid[labels == cluster].mean() if (labels == cluster).any() else 0.0
                  for cluster in range(k)]
    acid_cluster = int(np.argmax(acid_share))
    acid_cluster_descriptors = set(significant[significant.cluster == acid_cluster].tag)
    return dict(
        clusters_with_descriptor=int(significant.cluster.nunique()),
        significant_enrichments=len(significant),
        themes=themes,
        acid_cluster_pct_acid=100 * max(acid_share),
        acid_cluster_is_sour=bool(acid_cluster_descriptors & ACID_CLUSTER_DESCRIPTORS))


def extant_summary(matrix, inputs, n_init=6000):
    """Recompute the extant results of the paper on one binary call matrix.

    Parameters
    ----------
    matrix : ndarray of int
        Receptors x odorants, rows and columns as in `inputs.data`.
    inputs : ExtantInputs
    n_init : int
        Random starts of the k-medoids clustering, as in the paper.

    Returns
    -------
    dict
        One entry per reported number. Percentages are of the molecules in each class set.
    """
    data = inputs.data
    summary = {"called_cells": int(matrix.sum()),
               "odorants_with_a_call": int((matrix.sum(0) > 0).sum())}

    # Tuning breadth (Fig. 2b)
    breadth = ac.tuning_breadth(matrix, data.receptor_ids, inputs.class_of)
    summary["median_ligands_class_I"] = float(np.median(breadth["class_I"]))
    summary["median_ligands_class_II"] = float(np.median(breadth["class_II"]))
    summary["breadth_P"] = breadth["p_all"]

    # Which class reads which chemistry (Fig. 2c), with the 1/62 rule
    groups, _, _, _, _, _ = ac.class_partition(matrix, data.receptor_ids, inputs.class_of)
    for set_name, short in [("Class I only", "class_I_only"), ("shared", "shared"),
                            ("Class II only", "class_II_only")]:
        members = groups == set_name
        summary[f"n_{short}"] = int(members.sum())
        summary[f"pct_acid_{short}"] = 100 * inputs.is_acid[members].mean() if members.any() else np.nan
        summary[f"pct_aromatic_{short}"] = (100 * inputs.is_aromatic[members].mean()
                                            if members.any() else np.nan)
    only_I = groups == "Class I only"
    only_II = groups == "Class II only"
    acid_table = [[int(inputs.is_acid[only_I].sum()), int((~inputs.is_acid[only_I]).sum())],
                  [int(inputs.is_acid[only_II].sum()), int((~inputs.is_acid[only_II]).sum())]]
    summary["acid_class_I_only_vs_class_II_only_P"] = fisher_exact(acid_table)[1]

    # The six activation clusters (Fig. 2e, Table 1)
    bound = matrix.sum(0) > 0
    clustered_ids = inputs.odorant_ids[bound]
    distance = ac.jaccard_distance_between_columns(matrix[:, bound])
    _, labels = ac.kmedoids(distance, ac.N_CLUSTERS, n_init=n_init, seed=0)
    perception = cluster_perception(clustered_ids, labels, inputs.tags_by_odorant,
                                    inputs.is_acid[bound])
    perception["themes"] = "; ".join(sorted(perception["themes"]))
    summary.update(perception)
    in_paper = [i for i, odorant in enumerate(clustered_ids) if odorant in inputs.main_cluster_of]
    summary["adjusted_rand_index_vs_paper"] = adjusted_rand_score(
        [inputs.main_cluster_of[clustered_ids[i]] for i in in_paper], labels[in_paper])
    return summary


# ---------------------------------------------------------------------------------------------
# Ancestral receptors
# ---------------------------------------------------------------------------------------------

@dataclass
class AncestralInputs:
    """What the ancestral summary needs besides the call matrix."""

    ancestral_scores: pd.DataFrame   # 432 reconstructed nodes x odorants, tree labels
    extant_scores: pd.DataFrame      # 433 human receptors x the same odorants
    calibration_map: object          # the one map of calibration.py
    panel: list                      # SMILES of every odorant on the panel
    panel_groups: pd.DataFrame       # functional groups of the panel, indexed by SMILES
    times: object                    # ultrametric node times of ancestral_sets.mpl_times
    paper_binary: pd.DataFrame       # the paper's calls, to restore afterwards


def load_ancestral_inputs():
    """Load the ancestral and extant median predictions, aligned to the same odorants."""
    paper_binary, _, _, _ = an.density_grid()
    odorant_ids = list(paper_binary.columns)

    ancestral_scores = pd.read_csv(an.ASR_MATRIX, index_col=0)
    ancestral_scores.index = [str(name).replace("_ASR", "") for name in ancestral_scores.index]
    extant_scores = pd.read_csv(an.REF_MATRIX, index_col=0)
    extant_scores = extant_scores[[str(name).startswith("9606") for name in extant_scores.index]]

    pairs = pd.read_csv(an.PAIRS, usecols=["smiles_id", "SMILES"]).drop_duplicates("smiles_id")
    smiles_of = dict(zip(pairs.smiles_id, pairs.SMILES))
    panel = sorted({smiles_of[odorant] for odorant in odorant_ids if odorant in smiles_of})
    return AncestralInputs(
        ancestral_scores=ancestral_scores[odorant_ids], extant_scores=extant_scores[odorant_ids],
        calibration_map=reference_calibration_map(an.REF_MATRIX, an.PAIRS, an.FASTA),
        panel=panel, panel_groups=an.functional_groups(panel).set_index("SMILES"),
        times=an.mpl_times(), paper_binary=paper_binary)


def ancestral_calls(ancestral_calls_array, extant_calls_array, inputs):
    """Put ancestral and extant calls into one table indexed by tree node name."""
    odorant_ids = list(inputs.ancestral_scores.columns)
    binary = pd.concat([
        pd.DataFrame(ancestral_calls_array.astype(bool), index=inputs.ancestral_scores.index,
                     columns=odorant_ids),
        pd.DataFrame(extant_calls_array.astype(bool), index=inputs.extant_scores.index,
                     columns=odorant_ids)])
    return binary[~binary.index.duplicated()]


def _exponential_decay(distance, amplitude, rate, floor):
    return amplitude * np.exp(-rate * distance) + floor


def _use_calls(binary, inputs):
    """Make every function of ancestral_sets read `binary` instead of the paper's calls.

    ancestral_sets builds the paper's calls once and caches them (`density_grid`); every node
    set, edge table and trajectory is read from that cache. Replacing the cached call table is
    therefore the one place where another calling rule can enter, and it leaves the module's
    code untouched. `ancestral_summary` puts the paper's calls back when it is done.
    """
    _, probabilities, fills, ancestral_names = an.density_grid()
    an._DENSITY_CACHE["grid"] = (binary, probabilities, fills, ancestral_names)


def ancestral_summary(binary, inputs):
    """Recompute the ancestral results of the paper on one table of calls.

    Parameters
    ----------
    binary : DataFrame of bool
        Calls of the reconstructed nodes and the extant human receptors, indexed by tree name,
        from `ancestral_calls`. No measurements are written over the extant rows, as in the paper.
    inputs : AncestralInputs

    Returns
    -------
    dict
        Node sizes use the paper's numbering (node k is tree node_{k+1}).
    """
    _use_calls(binary, inputs)
    try:
        node_sets = an.node_sets([1, 2, 3, 6])
        summary = {f"node_{an.figure_node(n)}_ligands": len(node_sets[n]) for n in [1, 2, 3, 6]}

        # The first duplication: what each class ancestor kept, lost and gained (Fig. 3d)
        for child in [2, 3]:
            parent_set, child_set = node_sets[1], node_sets[child]
            branch = f"node_0_to_{an.figure_node(child)}"
            summary[f"{branch}_kept"] = len(parent_set & child_set)
            summary[f"{branch}_lost"] = len(parent_set - child_set)
            summary[f"{branch}_gained"] = len(child_set - parent_set)

        # Chemistry gained along the two founding duplications (Fig. 3e, Extended Data Fig. 7)
        for label, (parent, child, group) in FOUNDING_DUPLICATIONS.items():
            parent_number, child_number = int(parent.split("_")[1]), int(child.split("_")[1])
            gained = sorted(node_sets[child_number] - node_sets[parent_number])
            background = [smiles for smiles in inputs.panel if smiles not in gained]
            with_group = int(inputs.panel_groups.loc[gained, group].sum()) if gained else 0
            background_with_group = int(inputs.panel_groups.loc[background, group].sum())
            odds_ratio, p_value = (fisher_exact([[with_group, len(gained) - with_group],
                                                 [background_with_group,
                                                  len(background) - background_with_group]])
                                   if gained else (np.nan, np.nan))
            short = "duplication_1_acid" if "acid" in group else "duplication_2_aromatic"
            summary[f"{short}_gained"] = f"{with_group} of {len(gained)}"
            summary[f"{short}_odds_ratio"] = odds_ratio
            summary[f"{short}_P"] = p_value

        # The combinatorial-coding trajectory (Fig. 3h), without measurements at the tips
        trajectory = an.trajectory(inputs.times, binary=binary)
        first_duplication = trajectory[trajectory.n_lineages >= 2].iloc[0]
        summary["combinatorial_at_first_duplication"] = (
            f"{int(first_duplication.n_combinatorial)} of {int(first_duplication.n_detected)}")
        majority = trajectory.n_combinatorial > trajectory.n_single
        summary["first_majority_at_lineages"] = (int(trajectory[majority].n_lineages.iloc[0])
                                                 if majority.any() else None)
        minority_after_split = trajectory[~majority & (trajectory.n_lineages >= 2)]
        if len(minority_after_split) == 0:
            summary["majority_from_lineages"] = 2
        else:
            later = trajectory[trajectory.index > minority_after_split.index.max()]
            summary["majority_from_lineages"] = int(later.n_lineages.iloc[0]) if len(later) else None
        summary["present_day_combinatorial_vs_single"] = (
            f"{int(trajectory.n_combinatorial.iloc[-1])} vs {int(trajectory.n_single.iloc[-1])}")

        # Decay of parent-child similarity with branch length (Fig. 3f)
        edges = an.edge_table(None)
        fit, _ = curve_fit(_exponential_decay, edges.branch_len, edges.similarity,
                           p0=[0.5, 5.0, 0.3], bounds=([0, 0, 0], [1.5, 1e3, 1.0]))
        summary["decay_branches"] = len(edges)
        summary["half_life"] = float(np.log(2) / fit[1])
        summary["asymptote"] = float(fit[2])

        # Reconstructed against extant breadth (Extended Data Fig. 6a)
        _, _, _, ancestral_names = an.density_grid()
        ligands = binary.sum(1)
        is_ancestor = binary.index.isin(list(ancestral_names))
        summary["median_ligands_ancestors"] = float(ligands[is_ancestor].median())
        summary["median_ligands_extant"] = float(ligands[~is_ancestor].median())
    finally:
        _use_calls(inputs.paper_binary, inputs)
    return summary


# ---------------------------------------------------------------------------------------------
# Clustering-free test of "similar barcodes smell alike"
# ---------------------------------------------------------------------------------------------

def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _partial(xy, xz, yz):
    return (xy - xz * yz) / np.sqrt((1 - xz ** 2) * (1 - yz ** 2))


def mantel_test(barcode_distance, descriptor_distance, structure_distance, n_permutations=999,
                seed=0):
    """Rank (Spearman) Mantel test between barcode and odour-descriptor distances.

    The partial test removes the part of both distances that structural distance explains,
    since the model reads structure and similar molecules can share both a barcode and a smell.
    Odorant labels of the barcode matrix are permuted; the other two stay fixed.

    Returns
    -------
    dict
        rho and P for the plain and the partial test, and the structure-descriptor correlation.
    """
    upper = np.triu_indices(len(barcode_distance), 1)
    barcode_rank = rankdata(barcode_distance[upper])
    descriptor_rank = rankdata(descriptor_distance[upper])
    structure_rank = rankdata(structure_distance[upper])

    structure_descriptor = _pearson(structure_rank, descriptor_rank)
    observed = _pearson(barcode_rank, descriptor_rank)
    observed_partial = _partial(observed, _pearson(barcode_rank, structure_rank),
                                structure_descriptor)

    rng = np.random.default_rng(seed)
    at_least_plain, at_least_partial = 0, 0
    for _ in range(n_permutations):
        order = rng.permutation(len(barcode_distance))
        permuted_rank = rankdata(barcode_distance[np.ix_(order, order)][upper])
        plain = _pearson(permuted_rank, descriptor_rank)
        partial = _partial(plain, _pearson(permuted_rank, structure_rank), structure_descriptor)
        at_least_plain += plain >= observed
        at_least_partial += partial >= observed_partial
    return dict(
        odorants=len(barcode_distance), pairs=len(upper[0]),
        rho_barcode_descriptor=observed, P=(at_least_plain + 1) / (n_permutations + 1),
        rho_partial_on_structure=observed_partial,
        P_partial=(at_least_partial + 1) / (n_permutations + 1),
        rho_structure_descriptor=structure_descriptor, permutations=n_permutations)


def barcode_perception_test(matrix, inputs, n_permutations=999):
    """Do odorants with similar receptor barcodes have similar odour descriptors on this matrix?

    The Mantel test of `mantel_test`, on the odorants that have at least one call in `matrix`
    and at least one odour descriptor. Distances: Jaccard between barcodes, Jaccard between
    descriptor sets, and 1 - Tanimoto between Morgan fingerprints (radius 2, 2,048 bits).

    Parameters
    ----------
    matrix : ndarray of int
        Receptors x odorants, columns as in `inputs.odorant_ids`.
    inputs : ExtantInputs
    n_permutations : int

    Returns
    -------
    dict
        The entries of `mantel_test`.
    """
    use = np.array([matrix[:, i].sum() > 0 and odorant in inputs.tags_by_odorant
                    for i, odorant in enumerate(inputs.odorant_ids)])
    odorants = inputs.odorant_ids[use]
    barcode_distance = ac.jaccard_distance_between_columns(matrix[:, use])

    descriptor_sets = [set(inputs.tags_by_odorant[odorant]) for odorant in odorants]
    n = len(odorants)
    descriptor_distance = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            union = len(descriptor_sets[i] | descriptor_sets[j])
            shared = len(descriptor_sets[i] & descriptor_sets[j])
            descriptor_distance[i, j] = descriptor_distance[j, i] = 1 - shared / union if union else 0.0

    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(inputs.smiles_by_odorant[odorant]), 2, 2048) for odorant in odorants]
    structure_distance = np.array([1 - np.array(DataStructs.BulkTanimotoSimilarity(fp, fingerprints))
                                   for fp in fingerprints])
    return mantel_test(barcode_distance, descriptor_distance, structure_distance, n_permutations)


# ---------------------------------------------------------------------------------------------
# Re-clustering on subsamples (run in worker processes by cluster_number)
# ---------------------------------------------------------------------------------------------

def recluster_subsample(task):
    """Cluster a random 80% of the odorants, or of the receptors, into six groups.

    Parameters
    ----------
    task : tuple
        (what, seed, matrix): `what` is "odorants" or "receptors", `matrix` the binary matrix
        restricted to odorants with at least one call.

    Returns
    -------
    tuple
        (what, seed, kept odorant indices, labels of those odorants)
    """
    what, seed, matrix = task
    rng = np.random.default_rng(seed)
    n_receptors, n_odorants = matrix.shape
    if what == "receptors":
        kept_receptors = np.sort(rng.choice(n_receptors, int(0.8 * n_receptors), replace=False))
        kept_odorants = np.arange(n_odorants)
        distance = ac.jaccard_distance_between_columns(matrix[kept_receptors])
    else:
        kept_odorants = np.sort(rng.choice(n_odorants, int(0.8 * n_odorants), replace=False))
        distance = ac.jaccard_distance_between_columns(matrix[:, kept_odorants])
    _, labels = ac.kmedoids(distance, ac.N_CLUSTERS, n_init=6000, seed=0)
    return what, seed, kept_odorants, labels
