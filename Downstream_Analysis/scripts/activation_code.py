"""The human receptor activation code: the numbers behind Figure 2 and its supplementary notebooks.

Everything here returns arrays and tables. The figures are drawn by `activation_figures.py`,
so a notebook reads as: load, compute, draw.

Used by `notebooks/reference_tree/figure2_activation_code.ipynb` and the notebooks in
`notebooks/reference_tree/supplementary/`.
"""

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2_contingency, hypergeom

from calibration import FillResult, density_fill, fit_calibration_map

# Volatility comes from the chemical-space pipeline, so the repository has one definition of it.
REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "Chemicals"))
from chemspace import boiling_point, log_vapour_pressure  # noqa: E402

HUMAN_PREFIX = "9606"
N_CLUSTERS = 6


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------

def read_fasta(path):
    """Read a FASTA file into {first word of the header: upper-case sequence}."""
    sequences = {}
    current_id = None
    chunks = []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(chunks).upper()
            current_id = line[1:].split()[0]
            chunks = []
        else:
            chunks.append(line)
    if current_id:
        sequences[current_id] = "".join(chunks).upper()
    return sequences


def binding_rate_by_threshold(scores, thresholds):
    """Fraction of `scores` strictly above each threshold."""
    return np.array([(scores > threshold).mean() for threshold in thresholds])


def rate_matched_cut(scores, labels):
    """The score at which the predicted binding rate equals the measured one.

    Used only to report per-pair classification performance. It never fills a matrix.

    Parameters
    ----------
    scores : array of float
        Model probabilities of the measured pairs.
    labels : array of int
        The measured responses, 0 or 1.

    Returns
    -------
    float
    """
    thresholds = np.arange(0.50, 0.9990, 0.0001)
    predicted_rate = binding_rate_by_threshold(scores, thresholds)
    return float(thresholds[np.abs(predicted_rate - labels.mean()).argmin()])


@dataclass
class ActivationData:
    """The human activation matrix and the measurements it was built from."""

    receptor_ids: list            # 433 tree tips, "9606.<UniProt>"
    odorant_ids: list             # 754 "SMLxxxx" ids
    scores: np.ndarray            # model probability, receptors x odorants
    measured: pd.DataFrame        # one row per measured pair: pid, smiles_id, Responsive
    measured_rows: np.ndarray     # row index of each measured pair in `scores`
    measured_cols: np.ndarray     # column index of each measured pair
    measured_labels: np.ndarray   # measured response, 0 or 1
    measured_scores: np.ndarray   # model probability of each measured pair
    tested: np.ndarray            # True where a measurement exists
    experimental: np.ndarray      # measured response on the grid, 0 where untested
    calibration_map: object       # isotonic map, score -> measured binding frequency
    rate_matched_cut: float       # reported only; builds nothing
    fill: FillResult              # the density fill with the measurements overlaid
    hybrid: np.ndarray            # the binary matrix every analysis uses


def load_activation_data(probabilities_path, pairs_path, fasta_path):
    """Build the hybrid matrix: measurements where they exist, the density fill elsewhere.

    Measured pairs are matched to receptors on exact amino-acid sequence, never on UniProt
    accession, so M2OR mutants (which share their parent's accession) drop out instead of
    being merged onto the wild type. Repeated measurements of one pair count as positive if
    any of them is.

    Parameters
    ----------
    probabilities_path, pairs_path, fasta_path : path
        The median prediction matrix, the M2OR pairs table and the reference-tree FASTA.

    Returns
    -------
    ActivationData
    """
    probabilities = pd.read_csv(probabilities_path, index_col=0)
    receptor_ids = [i for i in probabilities.index if str(i).startswith(HUMAN_PREFIX)]
    human = probabilities.loc[receptor_ids]
    odorant_ids = list(human.columns)
    scores = human.values

    # One sequence can belong to more than one tree tip; the first one takes the measurement.
    sequences = read_fasta(fasta_path)
    receptors_by_sequence = {}
    for receptor in receptor_ids:
        receptors_by_sequence.setdefault(sequences[receptor], []).append(receptor)

    pairs = pd.read_csv(pairs_path, usecols=["Species", "UniProt ID", "Sequence", "Responsive",
                                             "smiles_id", "mutation"])
    human_pairs = pairs[pairs["Species"].str.lower().str.contains("homo")].copy()
    human_pairs["Sequence"] = human_pairs["Sequence"].str.upper().str.strip()
    human_pairs["pid"] = human_pairs["Sequence"].map(
        lambda sequence: receptors_by_sequence.get(sequence, [None])[0])
    matched = human_pairs[human_pairs["pid"].notna()
                          & human_pairs["smiles_id"].isin(set(odorant_ids))]
    measured = matched.groupby(["pid", "smiles_id"])["Responsive"].max().reset_index()

    row_of = {receptor: i for i, receptor in enumerate(receptor_ids)}
    col_of = {odorant: j for j, odorant in enumerate(odorant_ids)}
    rows = np.array([row_of[p] for p in measured["pid"]])
    cols = np.array([col_of[c] for c in measured["smiles_id"]])
    labels = measured["Responsive"].values.astype(int)
    measured_scores = scores[rows, cols]

    tested = np.zeros(scores.shape, bool)
    tested[rows, cols] = True
    experimental = np.zeros(scores.shape, np.int8)
    experimental[rows, cols] = labels

    calibration_map = fit_calibration_map(measured_scores, labels)
    fill = density_fill(scores, calibration_map, tested_mask=tested, experimental=experimental)

    return ActivationData(
        receptor_ids=receptor_ids, odorant_ids=odorant_ids, scores=scores, measured=measured,
        measured_rows=rows, measured_cols=cols, measured_labels=labels,
        measured_scores=measured_scores, tested=tested, experimental=experimental,
        calibration_map=calibration_map, rate_matched_cut=rate_matched_cut(measured_scores, labels),
        fill=fill, hybrid=fill.matrix.astype(int))


def tested_pairs_table(probabilities_path, pairs_path, fasta_path):
    """Every measured human pair next to its model probability, for the model check.

    Joined on amino-acid sequence. Keeps only pairs with a measured label.

    Returns
    -------
    common_pairs : DataFrame
        One row per measured pair, without the binary call (added once the cut is chosen).
    probability_grid : DataFrame
        Probabilities on the full grid of common sequences x common odorants.
    human_probabilities : DataFrame
        The human rows of the wide prediction table, with sequence and UniProt id attached.
    human_pairs : DataFrame
        The human M2OR records.
    """
    pairs = pd.read_csv(pairs_path)
    probabilities = pd.read_csv(probabilities_path)
    fasta_sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}

    odorant_columns = [c for c in probabilities.columns if c.startswith("SML")]
    human_probabilities = probabilities[probabilities["protein_id"].str.startswith(HUMAN_PREFIX)].copy()
    human_probabilities["sequence"] = human_probabilities["protein_id"].map(fasta_sequences)
    human_probabilities["uniprot_id"] = human_probabilities["protein_id"].str.split(".").str[1]

    human_pairs = (pairs[pairs["Species"] == "homo sapiens"]
                   [["UniProt ID", "Sequence", "seq_id", "smiles_id", "SMILES", "Responsive",
                     "mutation", "Class"]]
                   .copy()
                   .rename(columns={"UniProt ID": "uniprot_id", "Sequence": "sequence"}))

    # A pair measured more than once counts as positive if any measurement is.
    measured_grid = human_pairs.pivot_table(index="sequence", columns="smiles_id",
                                            values="Responsive", aggfunc="max")

    common_odorants = sorted(set(odorant_columns) & set(measured_grid.columns))
    common_sequences = sorted(set(human_probabilities["sequence"].dropna()) & set(measured_grid.index))

    by_sequence = (human_probabilities.dropna(subset=["sequence"])
                   .drop_duplicates("sequence").set_index("sequence"))
    measured_grid = measured_grid.loc[common_sequences, common_odorants]
    probability_grid = by_sequence.loc[common_sequences, common_odorants]

    measured_long = measured_grid.reset_index().melt(id_vars="sequence", var_name="smiles_id",
                                                     value_name="experimental")
    probability_long = probability_grid.reset_index().melt(id_vars="sequence", var_name="smiles_id",
                                                           value_name="probability")
    common_pairs = measured_long.merge(probability_long, on=["sequence", "smiles_id"])
    common_pairs = common_pairs.dropna(subset=["experimental"]).copy()
    common_pairs["experimental"] = common_pairs["experimental"].astype(int)

    smiles_of = pairs[["smiles_id", "SMILES"]].drop_duplicates("smiles_id").set_index("smiles_id")["SMILES"]
    measured_info = human_pairs.drop_duplicates("sequence").set_index("sequence")[["seq_id", "mutation", "Class"]]
    predicted_info = human_probabilities.drop_duplicates("sequence").set_index("sequence")[["uniprot_id"]]

    common_pairs["SMILES"] = common_pairs["smiles_id"].map(smiles_of)
    common_pairs["seq_id"] = common_pairs["sequence"].map(measured_info["seq_id"])
    common_pairs["uniprot_id"] = common_pairs["sequence"].map(predicted_info["uniprot_id"])
    common_pairs["mutation"] = common_pairs["sequence"].map(measured_info["mutation"])
    common_pairs["OR_class"] = common_pairs["sequence"].map(measured_info["Class"])

    common_pairs = common_pairs[["uniprot_id", "seq_id", "sequence", "smiles_id", "SMILES",
                                 "experimental", "probability", "mutation", "OR_class"]].reset_index(drop=True)
    return common_pairs, probability_grid, human_probabilities, human_pairs


def odour_tags(pairs_path, odours_path, odorant_ids):
    """SMILES and odour descriptors for each panel odorant.

    Returns
    -------
    smiles_by_odorant : dict
        "SMLxxxx" -> SMILES.
    tags_by_odorant : dict
        "SMLxxxx" -> list of lower-case descriptors, only for odorants that have any.
    """
    pairs = pd.read_csv(pairs_path, usecols=["SMILES", "smiles_id"])
    smiles_by_odorant = pairs.drop_duplicates("smiles_id").set_index("smiles_id")["SMILES"].to_dict()
    odours = pd.read_csv(odours_path)
    tags_by_smiles = {smiles: [tag.strip().lower() for tag in text.split(",") if tag.strip()]
                      for smiles, text in zip(odours["SMILES"], odours["Odors"])
                      if isinstance(text, str) and text.strip()}
    tags_by_odorant = {odorant: tags_by_smiles[smiles_by_odorant[odorant]] for odorant in odorant_ids
                       if smiles_by_odorant.get(odorant) in tags_by_smiles}
    return smiles_by_odorant, tags_by_odorant


def receptor_classes(classes_path):
    """{tree tip: "I" or "II"} from the human class table."""
    table = pd.read_csv(classes_path)
    return {row.leaf_id: ("I" if row.OR_class.split("_")[1] == "I" else "II")
            for row in table.itertuples()}


def tree_tip_order(tree_path):
    """Tip names in the order they are written in the Newick file (the ladderised order)."""
    return re.findall(r"[\(,]([^(),:;]+):", open(tree_path).read())


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def benjamini_hochberg(p_values):
    """Benjamini-Hochberg q values, in the input order."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    q_ranked = np.minimum.accumulate((ranked * n / (np.arange(n) + 1))[::-1])[::-1]
    q_values = np.empty(n)
    q_values[order] = np.clip(q_ranked, 0, 1)
    return q_values


def significance_stars(q):
    """'***' below 0.001, '**' below 0.01, '*' below 0.05, otherwise ''."""
    return "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else ""


def cliffs_delta(a, b):
    """P(a > b) - P(a < b) over all pairs. Positive means the first group is larger."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.sign(a[:, None] - b[None, :]).mean())


def hodges_lehmann(a, b, n_boot=5000, seed=0):
    """Median pairwise difference a - b, with a bootstrap 95% CI.

    Returns
    -------
    estimate, ci_low, ci_high : float
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    estimate = float(np.median(a[:, None] - b[None, :]))
    rng = np.random.default_rng(seed)
    boot = [float(np.median(rng.choice(a, len(a))[:, None] - rng.choice(b, len(b))[None, :]))
            for _ in range(n_boot)]
    low, high = np.percentile(boot, [2.5, 97.5])
    return estimate, float(low), float(high)


def kmedoids(distance, k, n_init=6000, seed=0):
    """Partitioning around medoids, best of `n_init` random starts.

    6000 starts is where the partition stops changing here: across five seeds the cost
    spread is 0.169 at 1500 starts, 0.009 at 3000, and 0 from 6000 up.

    Returns
    -------
    cost : float
    labels : array of int
    """
    rng = np.random.default_rng(seed)
    n = len(distance)
    best = None
    for _ in range(n_init):
        medoids = rng.choice(n, k, replace=False)
        for _ in range(300):
            labels = distance[:, medoids].argmin(1)
            new_medoids = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members):
                    new_medoids[c] = members[distance[np.ix_(members, members)].sum(1).argmin()]
            if set(new_medoids) == set(medoids):
                medoids = new_medoids
                break
            medoids = new_medoids
        labels = distance[:, medoids].argmin(1)
        cost = distance[np.arange(n), medoids[labels]].sum()
        if best is None or cost < best[0]:
            best = (cost, labels.copy())
    return best


def pcoa(distance):
    """Principal coordinate analysis (classical MDS).

    Returns
    -------
    coordinates : array
        One column per positive eigenvalue.
    eigenvalues : array
        All eigenvalues, largest first. Negative ones measure how non-Euclidean the distance is.
    """
    distance = np.asarray(distance, float)
    n = len(distance)
    centring = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centring.dot(distance ** 2).dot(centring)
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 1e-9
    return eigenvectors[:, positive] * np.sqrt(eigenvalues[positive]), eigenvalues


def jaccard_distance_between_columns(matrix):
    """Square Jaccard distance between the columns of a binary matrix; 0/0 counts as 0."""
    return np.nan_to_num(squareform(pdist(matrix.T, "jaccard")))


# ---------------------------------------------------------------------------------------------
# The six activation clusters
# ---------------------------------------------------------------------------------------------

# Odour themes used to name and colour the clusters. The first theme whose descriptors are
# significantly enriched in a cluster, taken in order of fold enrichment, gives it its name.
THEME_RULES = [
    (("acidic", "sour", "cheesy", "dairy"), "Acid / Dairy / Cheese", "#543005"),
    (("chemical", "potato", "vanilla", "roasted", "phenolic", "popcorn", "spicy"),
     "Roasted / Phenolic / Spice", "#ae7121"),
    (("plum", "jasmine", "floral", "balsamic"), "Floral / Balsamic", "#e7cf94"),
    (("lavender", "violet", "green", "fresh", "herbal", "citrus"), "Fresh / Green / Herbal", "#98d7cd"),
    (("cognac", "coconut", "pineapple", "aldehydic", "fruity", "waxy", "buttery"),
     "Fruity / Fatty / Aldehydic", "#98d7cd"),
    (("alcoholic", "ethereal", "medicinal", "fermented", "smoky", "rummy", "solvent"),
     "Fermented / Solvent", "#24877f"),
    (("musk", "animal", "grapefruit"), "Musk / Animal", "#003c30"),
]

SHORT_THEME = {
    "Acid / Dairy / Cheese": "Acid·Dairy", "Roasted / Phenolic / Spice": "Roasted·Spice",
    "Floral / Balsamic": "Floral", "Fresh / Green / Herbal": "Fresh·Green",
    "Fruity / Fatty / Aldehydic": "Fruity·Fatty", "Fermented / Solvent": "Fermented",
    "Musk / Animal": "Musk·Animal",
}


def cluster_odorants(matrix, odorant_ids, k=N_CLUSTERS):
    """Cluster the odorants that bind at least one receptor on the Jaccard distance of their columns.

    Returns
    -------
    clustered_ids : array of str
        The odorants with at least one binding call.
    clustered_matrix : array
        `matrix` restricted to those columns.
    labels : array of int
        PAM cluster of each clustered odorant.
    """
    odorant_ids = np.array(odorant_ids)
    bound = matrix.sum(0) > 0
    clustered_ids = odorant_ids[bound]
    clustered_matrix = matrix[:, bound]
    _, labels = kmedoids(jaccard_distance_between_columns(clustered_matrix), k)
    return clustered_ids, clustered_matrix, labels


def tag_enrichment(clustered_ids, labels, tags_by_odorant, k=N_CLUSTERS):
    """Hypergeometric enrichment of every odour descriptor in every cluster.

    Only tagged odorants count, and only descriptors carried by at least five of them are
    tested. q values are Benjamini-Hochberg over all cluster x descriptor tests.

    Returns
    -------
    DataFrame
        One row per test: cluster, n, tag, obs, tag_total, exp, obs_exp, p, fdr.
    """
    tagged_ids = {odorant: tags_by_odorant[odorant] for odorant in clustered_ids
                  if odorant in tags_by_odorant}
    is_tagged = np.array([odorant in tagged_ids for odorant in clustered_ids])
    tag_totals = Counter([tag for odorant in clustered_ids[is_tagged] for tag in tagged_ids[odorant]])
    n_tagged = int(is_tagged.sum())
    tested_tags = [tag for tag, count in tag_totals.items() if count >= 5]

    rows = []
    for c in range(k):
        members = [odorant for odorant in clustered_ids[labels == c] if odorant in tagged_ids]
        n_members = len(members)
        member_counts = Counter([tag for odorant in members for tag in tagged_ids[odorant]])
        for tag in tested_tags:
            observed = member_counts.get(tag, 0)
            total = tag_totals[tag]
            expected = n_members * total / n_tagged
            fold = observed / expected if expected > 0 else 0
            p = hypergeom.sf(observed - 1, n_tagged, total, n_members) if observed > 0 else 1.0
            rows.append([c, n_members, tag, observed, total, round(expected, 3), round(fold, 3), p])
    enrichment = pd.DataFrame(rows, columns=["cluster", "n", "tag", "obs", "tag_total", "exp",
                                             "obs_exp", "p"])
    enrichment["fdr"] = benjamini_hochberg(enrichment["p"].values)
    return enrichment


def significant_enrichment(enrichment):
    """The enriched rows that survive correction: q < 0.05 and fold above 1."""
    return enrichment[(enrichment.fdr < 0.05) & (enrichment.obs_exp > 1)]


def name_clusters(enrichment, labels, k=N_CLUSTERS):
    """Give every cluster a theme name and a distinct colour from its enriched descriptors.

    Themes are handed out greedily, strongest enrichment first, so no two clusters share a
    colour. A cluster that matches no theme is named after its top descriptor.

    Returns
    -------
    info : dict
        cluster -> {"name", "color", "size"}.
    display_order : list of int
        Clusters in the order of the theme palette, which is the order they are drawn in.
    """
    significant = significant_enrichment(enrichment)
    score = {}
    for c in range(k):
        cluster_rows = significant[significant.cluster == c]
        for theme_index, (keys, _, _) in enumerate(THEME_RULES):
            hits = cluster_rows[cluster_rows.tag.isin(keys)]
            if len(hits):
                score[(c, theme_index)] = float(hits.obs_exp.max())

    info = {c: None for c in range(k)}
    used_colours = set()
    for (c, theme_index), _ in sorted(score.items(), key=lambda item: -item[1]):
        _, name, colour = THEME_RULES[theme_index]
        if info[c] is None and colour not in used_colours:
            info[c] = dict(name=name, color=colour, size=int((labels == c).sum()))
            used_colours.add(colour)

    palette = list(dict.fromkeys(colour for _, _, colour in THEME_RULES))
    for c in range(k):
        if info[c] is None:
            colour = next((x for x in palette if x not in used_colours), "#8a8a8a")
            top = significant[significant.cluster == c].sort_values("obs_exp", ascending=False).tag.tolist()
            info[c] = dict(name=top[0].capitalize() if top else f"cluster{c}", color=colour,
                           size=int((labels == c).sum()))
            used_colours.add(colour)

    palette_position = {colour: i for i, colour in enumerate(palette)}
    display_order = sorted(range(k), key=lambda c: palette_position.get(info[c]["color"], 99))
    return info, display_order


# ---------------------------------------------------------------------------------------------
# Chemistry
# ---------------------------------------------------------------------------------------------

# 'aromatic ring' is [a], any aromatic atom, so pyrazines, furans, pyridines and thiophenes
# count. The benzene-only pattern it replaced missed 50 of the 233 aromatic odorants.
FUNCTIONAL_GROUPS = {
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "alcohol": "[#6;!$([CX3]=O)][OX2H1]",
    "ether": "[OD2]([#6])[#6]",
    "aromatic ring": "[a]",
    "amine": "[NX3;!$([NX3][CX3]=[OX1])]",
    "sulfur": "[#16]",
}
FUNCTIONAL_GROUP_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}

# The cluster descriptor panel, labelled property first so a figure reads without a glossary.
CLUSTER_DESCRIPTORS = [
    ("MolWt", "size (MW, Da)"),
    ("LogP", "greasiness (cLogP)"),
    ("TPSA", "polar surface (TPSA, Å$^2$)"),
    ("HBD", "H-bond donors (-OH, -NH)"),
    ("HBA", "H-bond acceptors"),
    ("OxygenAtoms", "oxygen atoms"),
    ("RotBonds", "flexibility (rotatable bonds)"),
    ("FracCSP3", "saturation (fraction sp$^3$)"),
    ("AromaticRings", "aromatic rings"),
    ("RingCount", "rings"),
]


def functional_group_flags(molecule):
    """{group: 1 if the molecule carries it, else 0}."""
    return {name: int(molecule.HasSubstructMatch(pattern))
            for name, pattern in FUNCTIONAL_GROUP_PATTERNS.items()}


def odorant_descriptors(odorant_ids, smiles_by_odorant, labels=None):
    """The cluster descriptor panel and functional-group flags, one row per parseable odorant."""
    records = []
    for i, odorant in enumerate(odorant_ids):
        smiles = smiles_by_odorant.get(odorant)
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            continue
        record = dict(sml=odorant, SMILES=smiles)
        if labels is not None:
            record["cluster"] = int(labels[i])
        record.update(
            MolWt=Descriptors.MolWt(molecule),
            LogP=Crippen.MolLogP(molecule),
            TPSA=rdMolDescriptors.CalcTPSA(molecule),
            HBD=Lipinski.NumHDonors(molecule),
            HBA=Lipinski.NumHAcceptors(molecule),
            OxygenAtoms=sum(atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()),
            RotBonds=Descriptors.NumRotatableBonds(molecule),
            FracCSP3=rdMolDescriptors.CalcFractionCSP3(molecule),
            AromaticRings=rdMolDescriptors.CalcNumAromaticRings(molecule),
            RingCount=rdMolDescriptors.CalcNumRings(molecule))
        record.update(functional_group_flags(molecule))
        records.append(record)
    return pd.DataFrame(records)


CLASS_SETS = ["Class I only", "shared", "Class II only"]


def class_partition(matrix, receptor_ids, class_of, rate=None):
    """Split the odorants by which receptor class reads them.

    An odorant is read by a class when it activates at least `rate` of that class's
    receptors. The default rate is 1/62, one Class I receptor or six Class II receptors,
    so the 62/371 difference in class size does not decide membership.

    Returns
    -------
    groups : array of str
        "Class I only", "shared", "Class II only", or "" for odorants neither class reads.
    rate : float
    rate_I, rate_II : array of float
        Fraction of each class activated by each odorant.
    in_I, in_II : array of bool
        Every odorant each class reads, shared ones included. These two sets overlap.
    """
    is_class_I = np.array([class_of[receptor] == "I" for receptor in receptor_ids])
    n_class_I = is_class_I.sum()
    n_class_II = (~is_class_I).sum()
    rate = 1.0 / n_class_I if rate is None else rate
    rate_I = matrix[is_class_I].sum(0) / n_class_I
    rate_II = matrix[~is_class_I].sum(0) / n_class_II
    in_I = rate_I >= rate
    in_II = rate_II >= rate
    groups = np.full(matrix.shape[1], "", dtype=object)
    groups[in_I & ~in_II] = "Class I only"
    groups[in_I & in_II] = "shared"
    groups[~in_I & in_II] = "Class II only"
    return groups, rate, rate_I, rate_II, in_I, in_II


def class_set_descriptors(odorant_ids, smiles_by_odorant, groups, rate_I, rate_II, in_I, in_II):
    """Descriptors, volatility and functional groups for every odorant a class reads.

    `WaterAff` is -cLogP, so a higher value means more at home in water, the same direction
    as TPSA and H-bond donors. `Tb` is the Joback boiling point and `VP` the log10 vapour
    pressure at 25 °C derived from it.
    """
    records = []
    for j, odorant in enumerate(odorant_ids):
        if not groups[j]:
            continue
        smiles = smiles_by_odorant.get(odorant)
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is None:
            continue
        tb = boiling_point(smiles)
        log_p = Crippen.MolLogP(molecule)
        record = dict(
            sml=odorant, group=groups[j], in_I=bool(in_I[j]), in_II=bool(in_II[j]),
            rate_I=rate_I[j], rate_II=rate_II[j],
            MolWt=Descriptors.MolWt(molecule), LogP=log_p, WaterAff=-log_p,
            TPSA=rdMolDescriptors.CalcTPSA(molecule),
            HBD=Lipinski.NumHDonors(molecule), HBA=Lipinski.NumHAcceptors(molecule),
            AromaticRings=rdMolDescriptors.CalcNumAromaticRings(molecule),
            RotBonds=Descriptors.NumRotatableBonds(molecule),
            RingCount=rdMolDescriptors.CalcNumRings(molecule),
            FracCSP3=rdMolDescriptors.CalcFractionCSP3(molecule),
            Heteroatoms=Lipinski.NumHeteroatoms(molecule),
            OxygenAtoms=sum(atom.GetAtomicNum() == 8 for atom in molecule.GetAtoms()),
            Tb=tb, VP=log_vapour_pressure(tb))
        record.update(functional_group_flags(molecule))
        records.append(record)
    return pd.DataFrame(records)


def receptors_reading(matrix, is_class_I, odorant_mask):
    """How many Class I and Class II receptors bind at least one odorant of the set."""
    subset = matrix[:, odorant_mask]
    return int((subset[is_class_I].sum(1) > 0).sum()), int((subset[~is_class_I].sum(1) > 0).sum())


def tag_percentages(members_I, members_II, odorant_ids, tags_by_odorant, min_pct=8.0, test=False):
    """Share of each set's tagged odorants carrying each odour descriptor.

    Percentages are over the odorants of the set that carry any descriptor. A descriptor is
    shown if it reaches `min_pct` in either set. With `test=True` each shown descriptor gets a
    Fisher test and Benjamini-Hochberg q over the shown ones; that needs two disjoint sets.

    Returns
    -------
    table : DataFrame
        Every descriptor, sorted by the gap between the two sets.
    shown : DataFrame
        The descriptors above `min_pct`, in the same order.
    n_tagged_I, n_tagged_II : int
    """
    from scipy.stats import fisher_exact

    odorant_ids = np.array(odorant_ids)
    tagged_I = [odorant for odorant in odorant_ids[members_I] if odorant in tags_by_odorant]
    tagged_II = [odorant for odorant in odorant_ids[members_II] if odorant in tags_by_odorant]
    counts_I = Counter(tag for odorant in tagged_I for tag in tags_by_odorant[odorant])
    counts_II = Counter(tag for odorant in tagged_II for tag in tags_by_odorant[odorant])
    n_tagged_I, n_tagged_II = len(tagged_I), len(tagged_II)

    rows = []
    # Sorted so rows with the same gap always come out in the same order.
    for tag in sorted(set(counts_I) | set(counts_II)):
        pct_I = 100 * counts_I[tag] / n_tagged_I
        pct_II = 100 * counts_II[tag] / n_tagged_II
        rows.append(dict(tag=tag, n_I=counts_I[tag], n_II=counts_II[tag], pct_I=pct_I,
                         pct_II=pct_II, gap=pct_I - pct_II))
    table = pd.DataFrame(rows).sort_values("gap", ascending=False, kind="stable").reset_index(drop=True)
    shown = table[(table.pct_I >= min_pct) | (table.pct_II >= min_pct)].reset_index(drop=True)

    if test:
        if (members_I & members_II).any():
            raise ValueError("per-tag Fisher needs disjoint sets; these overlap")
        p_values = [fisher_exact([[row.n_I, n_tagged_I - row.n_I], [row.n_II, n_tagged_II - row.n_II]])[1]
                    for row in shown.itertuples()]
        shown["p"] = p_values
        shown["q"] = benjamini_hochberg(np.array(p_values))
        shown["stars"] = [significance_stars(q) or "ns" for q in shown.q]
        table = table.merge(shown[["tag", "p", "q", "stars"]], on="tag", how="left")
    return table, shown, n_tagged_I, n_tagged_II


def tuning_breadth(matrix, receptor_ids, class_of):
    """Ligands per receptor by class, with the Mann-Whitney test, Cliff's delta and the shift.

    Orphan receptors (no ligand) are kept. The same statistics are also given with them
    removed, since the orphan share is similar in both classes and pulls them together.
    """
    from scipy.stats import mannwhitneyu

    is_class_I = np.array([class_of[receptor] == "I" for receptor in receptor_ids])
    n_ligands = matrix.sum(1)
    class_I, class_II = n_ligands[is_class_I], n_ligands[~is_class_I]
    class_I_bound, class_II_bound = class_I[class_I > 0], class_II[class_II > 0]
    table = pd.DataFrame({"pid": receptor_ids,
                          "OR_class": np.where(is_class_I, "Class I", "Class II"),
                          "n_ligands": n_ligands})
    return dict(
        table=table, is_class_I=is_class_I, class_I=class_I, class_II=class_II,
        p_all=mannwhitneyu(class_I, class_II).pvalue, delta_all=cliffs_delta(class_I, class_II),
        p_bound=mannwhitneyu(class_I_bound, class_II_bound).pvalue,
        delta_bound=cliffs_delta(class_I_bound, class_II_bound),
        shift_all=hodges_lehmann(class_I, class_II),
        shift_bound=hodges_lehmann(class_I_bound, class_II_bound),
        class_I_bound=class_I_bound, class_II_bound=class_II_bound)


# ---------------------------------------------------------------------------------------------
# Biosynthetic pathway
# ---------------------------------------------------------------------------------------------

# Polyketides (12) and carbohydrates (3) are left out. With all seven classes 20 of 42
# expected cells fall below 5 and chi-square is not valid; with these five it is 8 of 30.
PATHWAYS_ANALYSED = ["Fatty acids", "Shikimates and Phenylpropanoids", "Terpenoids",
                     "Alkaloids", "Amino acids and Peptides"]

PATHWAY_DESCRIPTORS = {
    "MolWt": Descriptors.MolWt, "LogP": Crippen.MolLogP, "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": Lipinski.NumHDonors, "HBA": Lipinski.NumHAcceptors,
    "AromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "RotBonds": Descriptors.NumRotatableBonds, "RingCount": rdMolDescriptors.CalcNumRings,
    "FracCSP3": rdMolDescriptors.CalcFractionCSP3, "Heteroatoms": Lipinski.NumHeteroatoms,
}


def build_ligand_pathways(coconut_csv, odours_path, out_path):
    """Rebuild `ligand_pathways.csv`: the 754 panel odorants joined to COCONUT on InChIKey.

    Only needed to regenerate the committed file, from the COCONUT bulk CSV
    (`coconut_csv-*.zip` from https://coconut.naturalproducts.net/download).
    """
    ligands = pd.read_csv(odours_path)[["InChIKey", "SMILES"]]
    keys = set(ligands["InChIKey"].dropna())
    columns = ["standard_inchi_key", "np_classifier_pathway", "np_classifier_superclass",
               "np_classifier_class", "name"]
    hits = [chunk[chunk["standard_inchi_key"].astype(str).isin(keys)]
            for chunk in pd.read_csv(coconut_csv, usecols=columns, chunksize=200_000, low_memory=False)]
    coconut = pd.concat(hits).drop_duplicates("standard_inchi_key").rename(columns={
        "standard_inchi_key": "InChIKey", "name": "coconut_name", "np_classifier_pathway": "pathway",
        "np_classifier_superclass": "superclass", "np_classifier_class": "npc_class"})
    frame = ligands.merge(coconut, on="InChIKey", how="left")
    frame["in_coconut"] = frame["pathway"].notna() | frame["coconut_name"].notna()
    frame.to_csv(out_path, index=False)
    return frame


def cramers_v(labels, categories):
    """Cramér's V between two labelings, with the chi-square p value and the table."""
    table = pd.crosstab(pd.Series(labels), pd.Series(categories))
    chi2, p, _, _ = chi2_contingency(table)
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), p, table


def permutation_floor(labels, categories, reps=2000, seed=0):
    """Mean and 95th percentile of Cramér's V with the labels shuffled.

    Not zero: a finite sample over skewed categories shows some association by chance.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    values = [cramers_v(rng.permutation(labels), categories)[0] for _ in range(reps)]
    return float(np.mean(values)), float(np.percentile(values, 95))


def pathway_association(clustered_ids, receptor_labels, smiles_by_odorant, odours_path,
                        pathways_path, k=N_CLUSTERS):
    """How well the activation clusters, and clusters built from descriptors alone, recover pathway.

    Pathway labels are the NPClassifier annotations COCONUT carries, joined on InChIKey,
    so only natural products get one.

    Returns
    -------
    dict
        pathway_all, pathway (only the analysed classes), use (odorants compared),
        descriptor_labels, results ({"receptor clusters" / "descriptor clusters": v, p, floor,
        floor95, ct}) and the recovery table written for the paper.
    """
    pathways = pd.read_csv(pathways_path)
    odours = pd.read_csv(odours_path)
    inchikey_by_smiles = dict(zip(odours["SMILES"], odours["InChIKey"]))
    pathway_by_inchikey = dict(zip(pathways["InChIKey"], pathways["pathway"]))

    pathway_all = np.array([pathway_by_inchikey.get(inchikey_by_smiles.get(smiles_by_odorant.get(odorant)), None)
                            for odorant in clustered_ids], dtype=object)
    pathway = np.array([p if p in PATHWAYS_ANALYSED else None for p in pathway_all], dtype=object)
    has_pathway = pd.notna(pathway)

    # The control: the same PAM on standardised descriptors, no receptors involved.
    rows = []
    for odorant in clustered_ids:
        molecule = Chem.MolFromSmiles(smiles_by_odorant.get(odorant) or "")
        rows.append([PATHWAY_DESCRIPTORS[name](molecule) if molecule else np.nan
                     for name in PATHWAY_DESCRIPTORS])
    descriptors = pd.DataFrame(rows, columns=list(PATHWAY_DESCRIPTORS), index=clustered_ids)
    parsed = descriptors.notna().all(1).values
    standardised = ((descriptors - descriptors.mean()) / descriptors.std()).values
    _, parsed_labels = kmedoids(squareform(pdist(standardised[parsed], "euclidean")), k)
    descriptor_labels = np.full(len(clustered_ids), -1)
    descriptor_labels[np.where(parsed)[0]] = parsed_labels

    use = has_pathway & (descriptor_labels >= 0)
    results = {}
    for name, labels in [("receptor clusters", receptor_labels), ("descriptor clusters", descriptor_labels)]:
        v, p, table = cramers_v(labels[use], pathway[use])
        floor, floor95 = permutation_floor(labels[use], pathway[use])
        results[name] = dict(v=v, p=p, floor=floor, floor95=floor95, ct=table)

    recovery = pd.DataFrame([{"clustering": name, "cramers_v": results[name]["v"],
                              "chi2_p": results[name]["p"], "floor": results[name]["floor"],
                              "floor_95th": results[name]["floor95"], "n": int(use.sum())}
                             for name in results])
    return dict(pathways=pathways, pathway_all=pathway_all, pathway=pathway, use=use,
                descriptor_labels=descriptor_labels, results=results, recovery=recovery)
