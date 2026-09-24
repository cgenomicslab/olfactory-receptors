"""Place the receptor panel's odorants on the map.

Rebuilds two groupings of the M2OR odorants from the receptor prediction matrix: six
co-tuning clusters (Jaccard k-medoids, k = 6) and the receptor class that reads each
odorant (class I only, both, class II only). Then measures whether the clusters overlap on
the map, and compares boiling point and NP-likeness across the groups (Kruskal-Wallis,
pairwise Mann-Whitney with BH, Cliff's delta).

Reads from Downstream_Analysis, Model_Inputs and Phylogenetic_Analysis.
Outputs in results/: panel_universe_join.csv, receptor_cluster_footprint.csv,
receptors_on_map_report.txt
"""
from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu

import chemspace as cs

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent
RESULTS = HERE / "results"

sys.path.append(str(REPOSITORY / "Downstream_Analysis" / "scripts"))
from calibration import density_fill, fit_calibration_map  # noqa: E402

PREDICTIONS = (REPOSITORY / "Downstream_Analysis" / "predictions" / "aggregated"
               / "Reference_Tree_Predictions" / "reference_median_probability_wide.csv")
PAIRS = REPOSITORY / "Model_Inputs" / "Data_Preparation" / "processed" / "m2or_pairs_model.csv"
SEQUENCES = (REPOSITORY / "Phylogenetic_Analysis" / "data" / "ReferenceTree"
             / "PF13853.9606_7955_7740_7764_75743_137246.fa")
CLASSES = REPOSITORY / "Phylogenetic_Analysis" / "data" / "HumanTree" / "human433_OR_classes.csv"

# The receptor analysis clusters odorants at k = 6, keeping the best of 6,000 random
# starts. Same values, so the clusters here are the same clusters.
N_CLUSTERS = 6
N_STARTS = 6000

CLASS_GROUPS = ["Class I only", "shared", "Class II only"]

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


# ------------------------------------------------------------- receptor side
def read_sequences(path):
    """Receptor id -> amino-acid sequence, from a FASTA file."""
    sequences, name, pieces = {}, None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                sequences[name] = "".join(pieces).upper()
            name, pieces = line[1:].split()[0], []
        else:
            pieces.append(line)
    if name:
        sequences[name] = "".join(pieces).upper()
    return sequences


def binary_response_matrix():
    """
    The human receptor x odorant matrix, as 0/1 responses.

    Predicted probabilities are calibrated against the measured M2OR responses and cut
    so the matrix has the density the measurements imply, with every measured pair set
    to its measured value (`calibration.density_fill`).

    Returns
    -------
    matrix : ndarray of int, (n_receptors, n_odorants)
    receptor_ids : list of str
    odorant_ids : list of str
        The smiles_id of each column.
    """
    predictions = pd.read_csv(PREDICTIONS, index_col=0)
    receptor_ids = [i for i in predictions.index if str(i).startswith("9606")]
    predicted = predictions.loc[receptor_ids]
    odorant_ids = list(predicted.columns)
    probabilities = predicted.values

    # M2OR identifies receptors by sequence; the prediction matrix by tree id.
    sequences = read_sequences(SEQUENCES)
    receptor_by_sequence = {}
    for receptor in receptor_ids:
        receptor_by_sequence.setdefault(sequences[receptor], receptor)

    measured = pd.read_csv(PAIRS, usecols=["Species", "Sequence", "Responsive", "smiles_id"])
    measured = measured[measured["Species"].str.lower().str.contains("homo")].copy()
    measured["receptor"] = (measured["Sequence"].str.upper().str.strip()
                            .map(receptor_by_sequence))
    measured = (measured[measured.receptor.notna()
                         & measured.smiles_id.isin(set(odorant_ids))]
                .groupby(["receptor", "smiles_id"])["Responsive"].max().reset_index())

    row_of = {r: i for i, r in enumerate(receptor_ids)}
    column_of = {c: j for j, c in enumerate(odorant_ids)}
    rows = np.array([row_of[r] for r in measured.receptor])
    columns = np.array([column_of[c] for c in measured.smiles_id])
    labels = measured["Responsive"].values.astype(int)

    tested = np.zeros(probabilities.shape, bool)
    tested[rows, columns] = True
    observed = np.zeros(probabilities.shape, np.int8)
    observed[rows, columns] = labels

    filled = density_fill(probabilities,
                          fit_calibration_map(probabilities[rows, columns], labels),
                          tested, observed)
    log(f"  receptors {len(receptor_ids)}, odorants {len(odorant_ids)}, "
        f"matrix density {filled.density:.4%}, cut {filled.cut:.4f}")
    return filled.matrix.astype(int), receptor_ids, odorant_ids


def k_medoids(distances, k, n_starts=N_STARTS, seed=0):
    """
    Partition around medoids, keeping the best of many random starts.

    Each start picks k medoids at random, then alternates between assigning every point
    to its nearest medoid and moving each medoid to the member with the smallest total
    distance to the rest of its cluster, until the medoids stop changing.

    Returns
    -------
    cost : float
        Total distance of every point to its medoid, for the best start.
    labels : ndarray of int
    """
    rng = np.random.default_rng(seed)
    n_points = len(distances)
    best = None
    for _ in range(n_starts):
        medoids = rng.choice(n_points, k, replace=False)
        for _ in range(300):
            labels = distances[:, medoids].argmin(1)
            moved = medoids.copy()
            for cluster in range(k):
                members = np.where(labels == cluster)[0]
                if len(members):
                    within = distances[np.ix_(members, members)].sum(1)
                    moved[cluster] = members[within.argmin()]
            if set(moved) == set(medoids):
                medoids = moved
                break
            medoids = moved
        labels = distances[:, medoids].argmin(1)
        cost = distances[np.arange(n_points), medoids[labels]].sum()
        if best is None or cost < best[0]:
            best = (cost, labels.copy())
    return best


def cotuning_clusters(matrix, odorant_ids):
    """Odorants clustered by which receptors read them (Jaccard, k-medoids)."""
    read_by_any = matrix.sum(0) > 0
    odorants = np.array(odorant_ids)[read_by_any]
    distances = np.nan_to_num(squareform(pdist(matrix[:, read_by_any].T, "jaccard")))
    cost, labels = k_medoids(distances, N_CLUSTERS)
    log(f"  co-tuning clusters: {len(odorants)} odorants read by at least one receptor, "
        f"cost {cost:.4f}, sizes {sorted(Counter(labels).values(), reverse=True)}")
    return odorants, labels, read_by_any


def class_groups(matrix, receptor_ids, read_by_any):
    """
    Which receptor class reads each odorant.

    An odorant counts as read by a class when at least 1/N of that class's receptors
    respond to it, where N is the number of class I receptors -- so "read by class I"
    means read by at least one class I receptor, and the same share is required of
    class II. Odorants below that share in both classes get no group.
    """
    classes = pd.read_csv(CLASSES)
    class_of = {row.leaf_id: ("I" if row.OR_class.split("_")[1] == "I" else "II")
                for row in classes.itertuples()}
    is_class_one = np.array([class_of[r] == "I" for r in receptor_ids])
    n_class_one, n_class_two = is_class_one.sum(), (~is_class_one).sum()

    share_one = matrix[is_class_one][:, read_by_any].sum(0) / n_class_one
    share_two = matrix[~is_class_one][:, read_by_any].sum(0) / n_class_two
    threshold = 1.0 / n_class_one

    group = np.where((share_one >= threshold) & (share_two < threshold), "Class I only",
            np.where((share_one >= threshold) & (share_two >= threshold), "shared",
            np.where((share_one < threshold) & (share_two >= threshold), "Class II only",
                     "")))
    counts = {g: int((group == g).sum()) for g in CLASS_GROUPS}
    log(f"  receptor classes: {n_class_one} class I, {n_class_two} class II; threshold "
        f"1/{n_class_one}; groups {counts}, below it in both {int((group == '').sum())}")
    return group


# ------------------------------------------------------------ chemical side
def join_to_universe(odorants, clusters, groups):
    """
    Give each clustered odorant its row in the universe.

    The universe is keyed by the InChIKey c02 computes from the desalted, canonical
    SMILES, so the panel's key is built the same way here rather than taken from M2OR.
    """
    smiles_of = (pd.read_csv(PAIRS, usecols=["SMILES", "smiles_id"])
                 .drop_duplicates("smiles_id").set_index("smiles_id")["SMILES"])

    rows = []
    for odorant, cluster, group in zip(odorants, clusters, groups):
        parent = cs.to_parent(smiles_of.get(odorant) or "")
        rows.append({"smiles_id": odorant, "cluster": int(cluster), "class_group": group,
                     "inchikey": parent[1] if parent else None})
    panel = pd.DataFrame(rows)

    universe = load_universe()
    positions = pd.DataFrame({"inchikey": universe.inchikey,
                              "row": np.arange(len(universe))})
    panel = panel.merge(positions, on="inchikey", how="left")
    panel["in_universe"] = panel["row"].notna()
    panel["row"] = panel["row"].astype("Int64")
    panel.to_csv(RESULTS / "panel_universe_join.csv", index=False)

    log(f"\n  clustered odorants              {len(panel)}")
    log(f"    InChIKey computed             {int(panel.inchikey.notna().sum())}")
    log(f"    found in the universe         {int(panel.in_universe.sum())}")
    on_map = panel[panel.in_universe].copy()
    for column in ["Tb_joback", "np_likeness"]:
        values = pd.to_numeric(universe[column].values[on_map.row.astype(int)],
                               errors="coerce")
        on_map[column] = values
        log(f"    with {column:24s} {int(np.isfinite(values).sum())}")
    coordinates = np.load(RESULTS / "umap.npy")
    on_map["x"] = coordinates[on_map.row.astype(int), 0]
    on_map["y"] = coordinates[on_map.row.astype(int), 1]
    return on_map, universe


def load_universe():
    """The universe, ordered to match the embedding and the map coordinates."""
    universe = pd.read_parquet(RESULTS / "universe.parquet",
                               columns=["inchikey", "Tb_joback", "np_likeness",
                                        "in_background"])
    index = pd.read_csv(RESULTS / "embed_index.csv")
    return universe.set_index("inchikey").loc[index.inchikey].reset_index()


# ---------------------------------------------------------------- questions
def question1_footprints(on_map):
    """Median map distance within each cluster, against the median to the other clusters."""
    log("\n" + "=" * 72)
    log("1  DO THE CLUSTERS SIT APART ON THE MAP, OR INTERLEAVE?")
    log("=" * 72)
    points = on_map[["x", "y"]].values
    clusters = on_map.cluster.values

    rows = []
    for cluster in range(N_CLUSTERS):
        inside = np.flatnonzero(clusters == cluster)
        outside = np.flatnonzero(clusters != cluster)
        within = np.median(pdist(points[inside])) if len(inside) > 1 else np.nan
        between = np.median(np.linalg.norm(
            points[inside][:, None, :] - points[outside][None, :, :], axis=2))
        rows.append({"cluster": cluster, "n": len(inside), "within": within,
                     "between": between, "ratio": within / between})
    footprint = pd.DataFrame(rows)
    footprint.to_csv(RESULTS / "receptor_cluster_footprint.csv", index=False)

    log("  median pairwise map distance; ratio near 1 = interleaved, well below 1 = apart")
    for _, row in footprint.iterrows():
        # row["between"], not row.between: that name is a pandas method.
        log(f"    cluster {int(row['cluster'])}  n={int(row['n']):3d}   "
            f"within {row['within']:5.2f}   between {row['between']:5.2f}   "
            f"ratio {row['ratio']:.2f}")
    log("  Descriptive only: map distances are not metric, so there is no test.")


def compare_groups(values_by_group, names):
    """Kruskal-Wallis across groups, then every pair (Mann-Whitney, Cliff's delta, BH)."""
    statistic, p_value = kruskal(*[v for v in values_by_group if len(v) >= 3])
    log(f"  Kruskal-Wallis H = {statistic:.1f}, P = {p_value:.2g}")
    for name, values in zip(names, values_by_group):
        log(f"    {name:14s} n={len(values):3d}   median {np.median(values):8.2f}")

    pairs = []
    for (i, a), (j, b) in combinations(enumerate(values_by_group), 2):
        pairs.append((names[i], names[j], mannwhitneyu(a, b).pvalue,
                      cs.cliffs_delta(a, b)))
    q_values = cs.benjamini_hochberg([p for _, _, p, _ in pairs])
    significant = [(pair, q) for pair, q in zip(pairs, q_values) if q < 0.05]
    log(f"  pairwise Mann-Whitney, BH across {len(pairs)} pairs: "
        f"{len(significant)} with q < 0.05")
    for (first, second, _, delta), q in sorted(significant, key=lambda t: t[1]):
        log(f"    {first} vs {second}: q = {q:.2g}, Cliff's delta = {delta:+.2f}")
    return statistic, p_value


def question2_cluster_properties(on_map):
    """Volatility and natural-product likeness across the six clusters."""
    log("\n" + "=" * 72)
    log("2  DO THE CLUSTERS DIFFER IN VOLATILITY OR NATURAL-PRODUCT LIKENESS?")
    log("=" * 72)
    names = [f"cluster {c}" for c in range(N_CLUSTERS)]
    for column, label in [("Tb_joback", "boiling point (K)"),
                          ("np_likeness", "natural-product likeness")]:
        log(f"\n  {label}")
        compare_groups([on_map.loc[on_map.cluster == c, column].dropna().values
                        for c in range(N_CLUSTERS)], names)


def question3_class_volatility(on_map, universe):
    """Volatility across the three receptor-class groups, and against other NPs."""
    log("\n" + "=" * 72)
    log("3  DO THE RECEPTOR-CLASS GROUPS DIFFER IN VOLATILITY?")
    log("=" * 72)
    values = [on_map.loc[on_map.class_group == g, "Tb_joback"].dropna().values
              for g in CLASS_GROUPS]
    compare_groups(values, CLASS_GROUPS)

    background = pd.to_numeric(universe.Tb_joback[universe.in_background == 1],
                               errors="coerce").dropna().values
    log(f"\n  against every other natural product (n = {len(background):,}, "
        f"median {np.median(background):.0f} K):")
    for group, group_values in zip(CLASS_GROUPS, values):
        log(f"    {group:14s} Cliff's delta {cs.cliffs_delta(group_values, background):+.2f}")


def main():
    RESULTS.mkdir(exist_ok=True)
    log("=" * 72)
    log("THE RECEPTOR PANEL ON THE NATURAL-PRODUCT MAP")
    log("=" * 72)

    matrix, receptor_ids, odorant_ids = binary_response_matrix()
    odorants, clusters, read_by_any = cotuning_clusters(matrix, odorant_ids)
    groups = class_groups(matrix, receptor_ids, read_by_any)
    on_map, universe = join_to_universe(odorants, clusters, groups)

    question1_footprints(on_map)
    question2_cluster_properties(on_map)
    question3_class_volatility(on_map, universe)

    (RESULTS / "receptors_on_map_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'receptors_on_map_report.txt'}")


if __name__ == "__main__":
    main()
