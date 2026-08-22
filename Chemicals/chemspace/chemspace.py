"""Shared chemistry and statistics helpers for the odorant chemical-space analysis.

The question this folder answers: odorants sit close together in chemical space, but is that
because odour chemistry is special, or only because odorants are small and evaporate easily?
The ChEMBL comparison could not tell those apart, because ChEMBL is mostly drug chemistry --
"odorants are not drugs" was doing much of the work. COCONUT (natural products) is the fairer
background, and these helpers are what the s01-s06 scripts share to build it.

Two conventions worth knowing, because everything downstream depends on them:

  * Molecules are joined on an InChIKey that RDKit recomputes from the SMILES, never on a
    COCONUT identifier. A database release bump therefore cannot silently break a join.
  * Every call that touches a GPU is guarded, so only s02 (and optionally s06) needs one.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, inchi, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

# Atomic numbers of F, Cl, Br, I.
HALOGENS = {9, 17, 35, 53}

# Gas constant, J / (mol K). Used by the vapour-pressure estimate below.
R_GAS = 8.314

# The descriptor panel. These are the 22 numbers that stand in for "what kind of molecule
# is this" whenever we are not using the MolFormer embedding.
DESCRIPTOR_NAMES = [
    "MolWt", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "AromaticRings", "RingCount",
    "AliphaticRings", "FracCsp3", "HeavyAtoms", "MolMR", "LabuteASA", "BertzCT",
    "NumStereo", "QED", "nHeteroatoms", "nO", "nN", "nS", "nHalogen", "nC",
]

# Functional groups, named the way a perfumer would name them rather than the way a
# cheminformatician would. Each value is a SMARTS pattern RDKit matches against a molecule.
FUNCTIONAL_GROUP_SMARTS = {
    "fg_alcohol": "[CX4;!$(C=O)][OX2H]",
    "fg_phenol": "[c][OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2H0][#6]",
    "fg_lactone": "[#6][OX2][CX3](=O)[#6]",
    "fg_ether": "[OD2]([#6])[#6]",
    "fg_aldehyde": "[CX3H1](=O)[#6]",
    "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_amine": "[NX3;H2,H1,H0;!$(NC=O);!$(N[a])]",
    "fg_amide": "[NX3][CX3](=[OX1])",
    "fg_thiol": "[#16X2H]",
    "fg_sulfide": "[#16X2H0][#6]",
    "fg_nitrile": "[NX1]#[CX2]",
    "fg_halogen": "[F,Cl,Br,I]",
    "fg_benzene": "c1ccccc1",
    "fg_terpene_isopropyl": "CC(C)[!#1]",
}

# Compiled once at import; matching against a compiled pattern is much faster.
FUNCTIONAL_GROUP_PATTERNS = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()
}
FUNCTIONAL_GROUP_NAMES = list(FUNCTIONAL_GROUP_SMARTS)

# Short aliases, kept because the scripts read better with them.
DESC = DESCRIPTOR_NAMES
FG_COLS = FUNCTIONAL_GROUP_NAMES
FG_SMARTS = FUNCTIONAL_GROUP_SMARTS
FG_PATTS = FUNCTIONAL_GROUP_PATTERNS


def device():
    """
    Return "cuda" if a GPU is visible, otherwise "cpu".

    Everything except s02 works either way; a GPU only makes it faster.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def free_gpu_memory():
    """Release cached GPU memory, if there is a GPU. A no-op on CPU."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------- chemistry
def largest_fragment_smiles(smiles):
    """
    Desalt a SMILES by keeping only its biggest piece.

    A SMILES with dots in it describes several disconnected pieces, usually a molecule
    plus a counter-ion ("CC(=O)[O-].[Na+]"). The chemistry we care about is in the
    largest piece, so that is the one we keep.

    Parameters
    ----------
    smiles : str
        Possibly multi-fragment SMILES.

    Returns
    -------
    str or None
        The largest fragment, or None if nothing could be parsed.
    """
    if "." not in smiles:
        return smiles

    largest = None
    largest_size = -1
    for fragment in smiles.split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            continue
        size = molecule.GetNumHeavyAtoms()
        if size > largest_size:
            largest = fragment
            largest_size = size
    return largest


def to_parent(smiles):
    """
    Turn a raw SMILES into the canonical form we join on.

    Desalts first, then asks RDKit for a canonical SMILES and an InChIKey. Recomputing
    the InChIKey ourselves, rather than trusting the one a database ships, is what keeps
    joins stable across COCONUT releases.

    Parameters
    ----------
    smiles : str
        Raw SMILES, as it appears in the source file.

    Returns
    -------
    tuple of (str, str), or None
        (canonical SMILES, InChIKey), or None if the SMILES could not be parsed.
    """
    if not isinstance(smiles, str) or not smiles:
        return None

    smiles = largest_fragment_smiles(smiles)
    if smiles is None:
        return None

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    try:
        canonical = Chem.MolToSmiles(molecule, isomericSmiles=True)
        inchikey = inchi.MolToInchiKey(molecule)
    except Exception:
        return None

    if not inchikey:
        return None
    return canonical, inchikey


def boiling_point(smiles):
    """
    Estimate a molecule's normal boiling point, in kelvin, by Joback's method.

    Joback's method counts the chemical groups in a molecule (how many -OH, how many
    ring carbons, and so on) and adds up a fitted contribution for each. We use it as a
    stand-in for volatility: the lower the boiling point, the more readily the molecule
    evaporates, and a molecule has to evaporate before it can be smelled.

    Two caveats that matter when reading any result built on this:

      * It is an estimate, not a measurement.
      * It is a linear sum, so it drifts badly for large, heavily-decorated molecules --
        many natural products come back with boiling points above their decomposition
        temperature. It is trustworthy in the small-molecule range where odorants live.

    Parameters
    ----------
    smiles : str
        Canonical SMILES.

    Returns
    -------
    float
        Boiling point in kelvin, or nan when the estimate fails or is unphysical.
    """
    try:
        from thermo import Joback

        estimator = Joback(smiles)
        group_counts = estimator.counts
        if not group_counts:
            return np.nan

        tb = estimator.Tb(group_counts)
        if tb is None or not np.isfinite(tb) or tb <= 0 or tb > 2000:
            return np.nan
        return float(tb)
    except Exception:
        return np.nan


def log_vapour_pressure(tb, temperature=298.15):
    """
    Estimate log10 of the vapour pressure, in atmospheres, at room temperature.

    Uses Clausius-Clapeyron with Trouton's rule, which approximates the heat of
    vaporisation as 88 * Tb. This is volatility expressed the way a perfumer would
    think about it -- how much of the molecule is in the air -- rather than as a
    boiling point.

    Parameters
    ----------
    tb : float
        Normal boiling point in kelvin, from `boiling_point`.
    temperature : float, optional
        Temperature in kelvin. Defaults to 25 C.

    Returns
    -------
    float
        log10(vapour pressure / atm), or nan if `tb` is nan.
    """
    if not np.isfinite(tb):
        return np.nan

    enthalpy_of_vaporisation = 88.0 * tb          # Trouton's rule
    exponent = -(enthalpy_of_vaporisation / R_GAS) * (1.0 / temperature - 1.0 / tb)
    return exponent / np.log(10.0)


def descriptors(smiles):
    """
    Compute everything we know about one molecule from its structure.

    That is: the 22-descriptor panel, a count for each of the 16 functional groups, the
    Joback boiling point, and the vapour pressure derived from it.

    Parameters
    ----------
    smiles : str
        Canonical SMILES.

    Returns
    -------
    dict or None
        One key per descriptor, functional group, plus "Tb_joback" and "logP_vap".
        None if the molecule could not be parsed.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None

    values = {}
    try:
        values["MolWt"] = Descriptors.MolWt(molecule)
        values["LogP"] = Crippen.MolLogP(molecule)
        values["TPSA"] = rdMolDescriptors.CalcTPSA(molecule)
        values["HBD"] = rdMolDescriptors.CalcNumHBD(molecule)
        values["HBA"] = rdMolDescriptors.CalcNumHBA(molecule)
        values["RotBonds"] = rdMolDescriptors.CalcNumRotatableBonds(molecule)
        values["AromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(molecule)
        values["RingCount"] = rdMolDescriptors.CalcNumRings(molecule)
        values["AliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(molecule)
        values["FracCsp3"] = rdMolDescriptors.CalcFractionCSP3(molecule)
        values["HeavyAtoms"] = molecule.GetNumHeavyAtoms()
        values["MolMR"] = Crippen.MolMR(molecule)
        values["LabuteASA"] = rdMolDescriptors.CalcLabuteASA(molecule)
        values["BertzCT"] = Descriptors.BertzCT(molecule)
        values["NumStereo"] = rdMolDescriptors.CalcNumAtomStereoCenters(molecule)
        values["QED"] = QED.qed(molecule)
        values["nHeteroatoms"] = rdMolDescriptors.CalcNumHeteroatoms(molecule)
    except Exception:
        return None

    # Count atoms of each element we care about, in one pass over the molecule.
    n_oxygen = n_nitrogen = n_sulfur = n_halogen = n_carbon = 0
    for atom in molecule.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        if atomic_number == 8:
            n_oxygen += 1
        elif atomic_number == 7:
            n_nitrogen += 1
        elif atomic_number == 16:
            n_sulfur += 1
        elif atomic_number == 6:
            n_carbon += 1
        elif atomic_number in HALOGENS:
            n_halogen += 1

    values["nO"] = n_oxygen
    values["nN"] = n_nitrogen
    values["nS"] = n_sulfur
    values["nHalogen"] = n_halogen
    values["nC"] = n_carbon

    for name, pattern in FUNCTIONAL_GROUP_PATTERNS.items():
        values[name] = len(molecule.GetSubstructMatches(pattern)) if pattern else 0

    values["Tb_joback"] = boiling_point(smiles)
    values["logP_vap"] = log_vapour_pressure(values["Tb_joback"])
    return values


def process(smiles):
    """
    Canonicalise one molecule and describe it, in a single call.

    This is what the multiprocessing pools in s01 map over.

    Returns
    -------
    tuple of (str, str, dict), or None
        (canonical SMILES, InChIKey, descriptor dict), or None on failure.
    """
    parent = to_parent(smiles)
    if parent is None:
        return None

    canonical, inchikey = parent
    values = descriptors(canonical)
    if values is None:
        return None
    return canonical, inchikey, values


# --------------------------------------------------------------------- statistics
def smd(group_a, group_b):
    """
    Standardised mean difference between two groups.

    The difference between the two means, divided by their pooled spread, so the answer
    has no units and is comparable across descriptors. We use it to check that matching
    worked: after matching, a value near zero means the two groups really are alike on
    that descriptor.

    Parameters
    ----------
    group_a, group_b : array_like
        The two sets of values to compare.

    Returns
    -------
    float
        The standardised mean difference, or 0.0 if neither group varies at all.
    """
    pooled_sd = np.sqrt((np.nanvar(group_a) + np.nanvar(group_b)) / 2)
    if pooled_sd <= 0:
        return 0.0
    return (np.nanmean(group_a) - np.nanmean(group_b)) / pooled_sd


def greedy_match(covariates, odorant_rows, background_rows, caliper_sd=0.25, seed=0):
    """
    Pair each odorant with the most similar unused natural product.

    Walks the odorants in a random order and gives each one its nearest unused partner,
    measured on the z-scored covariates. A partner further away than the caliper is
    refused, so an odorant with no good match stays unmatched rather than being paired
    with something unlike it. No natural product is used twice.

    Parameters
    ----------
    covariates : ndarray, shape (n_molecules, n_covariates)
        Z-scored covariates for every molecule in the universe.
    odorant_rows, background_rows : ndarray of int
        Row numbers of the odorants and of the natural products to draw partners from.
    caliper_sd : float, optional
        Largest allowed distance, in standard deviations per covariate.
    seed : int, optional
        Controls the order odorants are matched in. Different seeds give slightly
        different pairings, which is why s06 repeats this.

    Returns
    -------
    matched_odorants, matched_background : ndarray of int
        Row numbers of the pairs, aligned with each other.
    fraction_matched : float
        Share of odorants that found a partner.
    """
    import torch

    dev = device()
    rng = np.random.default_rng(seed)

    # The caliper is per-covariate, so scale it to the distance in the full space.
    caliper = caliper_sd * np.sqrt(covariates.shape[1])

    background = torch.from_numpy(covariates[background_rows]).float().to(dev)
    odorants = torch.from_numpy(covariates[odorant_rows]).float().to(dev)
    already_used = torch.zeros(len(background_rows), dtype=torch.bool, device=dev)

    partner_of = np.full(len(odorant_rows), -1, int)
    order = rng.permutation(len(odorant_rows))

    # Distances are computed a block at a time; the whole matrix would not fit in memory.
    for start in range(0, len(order), 256):
        block = order[start:start + 256]
        distances = torch.cdist(odorants[block], background)
        distances[:, already_used] = float("inf")

        for row_in_block, odorant in enumerate(block):
            nearest = int(torch.argmin(distances[row_in_block]).item())
            distance = float(distances[row_in_block, nearest].item())
            if np.isfinite(distance) and distance <= caliper:
                partner_of[odorant] = nearest
                already_used[nearest] = True
                # Stop this partner being offered again inside the same block.
                distances[:, nearest] = float("inf")

    del background, odorants
    free_gpu_memory()

    found = partner_of >= 0
    return odorant_rows[found], background_rows[partner_of[found]], float(found.mean())


def match_on(table, columns, odorant_rows, background_rows, caliper_sd=0.25, seed=0):
    """
    Match odorants to natural products on the named columns.

    Z-scores the columns first so that, say, molecular weight and logP count equally,
    then hands over to `greedy_match`. Molecules missing any of the columns are dropped.

    Parameters
    ----------
    table : DataFrame
        The universe, one row per molecule.
    columns : list of str
        Columns to match on, e.g. ["Tb_joback", "MolWt", "LogP"].
    odorant_rows, background_rows : ndarray of int
        Row numbers of the two groups.
    caliper_sd, seed
        Passed through to `greedy_match`.

    Returns
    -------
    Same as `greedy_match`.
    """
    values = table[columns].values.astype(float)
    complete = np.isfinite(values).all(axis=1)

    mean = np.nanmean(values[complete], axis=0)
    spread = np.nanstd(values[complete], axis=0)
    z_scored = (values - mean) / spread

    return greedy_match(
        z_scored,
        odorant_rows[complete[odorant_rows]],
        background_rows[complete[background_rows]],
        caliper_sd,
        seed,
    )


def knn(normalised_embeddings, query_rows, k=50, chunk=256, target_rows=None):
    """
    Find each query molecule's k nearest neighbours by cosine similarity.

    Exact, not approximate: every query is compared against every target. A molecule is
    never returned as its own neighbour.

    Parameters
    ----------
    normalised_embeddings : ndarray, shape (n_molecules, 768)
        Embeddings scaled to unit length, so a dot product is a cosine similarity.
    query_rows : ndarray of int
        Molecules to find neighbours for.
    k : int, optional
        Neighbours per query.
    chunk : int, optional
        Queries per block. Lower this if the GPU runs out of memory.
    target_rows : ndarray of int, optional
        Molecules allowed to be neighbours. Defaults to the whole universe.

    Returns
    -------
    neighbour_indices : ndarray, shape (n_queries, k)
        Row numbers, as positions within `target_rows`.
    similarities : ndarray, shape (n_queries, k)
        Cosine similarity of each neighbour.
    """
    import torch

    dev = device()
    targets = np.arange(len(normalised_embeddings)) if target_rows is None else target_rows
    target_matrix = torch.from_numpy(normalised_embeddings[targets]).to(dev)

    # For each universe row, where it sits inside `targets` (-1 if it is not a target).
    position_in_targets = np.full(len(normalised_embeddings), -1, np.int64)
    position_in_targets[targets] = np.arange(len(targets))

    all_indices = []
    all_similarities = []
    for start in range(0, len(query_rows), chunk):
        block = query_rows[start:start + chunk]
        similarity = torch.from_numpy(normalised_embeddings[block]).to(dev) @ target_matrix.T

        # Blank out self-matches, so a molecule is not its own nearest neighbour.
        own_position = position_in_targets[block]
        is_target = np.flatnonzero(own_position >= 0)
        if len(is_target):
            similarity[
                torch.from_numpy(is_target).to(dev),
                torch.from_numpy(own_position[is_target]).to(dev),
            ] = -2.0

        top_similarity, top_index = torch.topk(similarity, k, dim=1)
        all_indices.append(top_index.cpu().numpy())
        all_similarities.append(top_similarity.cpu().numpy())

    del target_matrix
    free_gpu_memory()
    return np.vstack(all_indices), np.vstack(all_similarities)


def benjamini_hochberg(p_values):
    """
    Convert p-values to q-values, controlling the false discovery rate.

    We test many functional groups and pathways at once, so some will look significant
    by chance. A q-value of 0.05 means that among everything called significant at that
    threshold, about 5% are expected to be false.

    Parameters
    ----------
    p_values : array_like
        One p-value per test.

    Returns
    -------
    ndarray
        q-values, in the same order as the input.
    """
    p_values = np.asarray(p_values, float)
    n_tests = len(p_values)

    order = np.argsort(p_values)
    scaled = p_values[order] * n_tests / (np.arange(n_tests) + 1)

    # Walking backwards and keeping the running minimum enforces monotonicity.
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]

    q_values = np.empty(n_tests)
    q_values[order] = np.clip(adjusted, 0, 1)
    return q_values
