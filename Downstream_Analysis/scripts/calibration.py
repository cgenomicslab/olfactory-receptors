"""Density-calibrated, rank-based binarisation of OR-odorant binding predictions.

One calibration map, defined once here and shared by the extant (reference-tree)
and ancestral notebooks, so both binarise on identical terms.

Why not a classification threshold
----------------------------------
The rate-matched cut (~0.915) is the operating point at which the predicted
binding proportion equals the measured one on the tested cells. It is the right
number to *report* for per-pair classification performance, and it is what the
confusion matrix and precision/recall in `figure2_activation_code` (Part 1) are computed at.

It is the wrong tool for filling a grid. Recall there is ~0.6, so thresholding
the full matrix keeps only the cells the model is most confident about and drops
roughly 40% of real binders. The resulting matrix is systematically under-filled.

What this module does instead
-----------------------------
Separate the two questions. *How many* cells should be on is answered by
calibration; *which* cells are on is answered by rank.

1. Isotonic regression maps model score -> empirically measured binding
   frequency, fitted once on the tested cells only.
2. That map is applied to every untested cell, giving each a calibrated
   probability of being a real binder.
3. Their sum is the expected number of true binders among the untested cells
   (linearity of expectation: E[sum X_i] = sum p_i). No threshold is involved.
4. The untested cells are ranked by model score and the top N are set to 1.
   The score of the Nth cell is the density cut. It is an *output* of the
   procedure, never a chosen input.
5. Where a measurement exists it is written over the fill. A measured value
   always beats a predicted one.

Assumption
----------
At a fixed model score, tested and untested pairs bind at the same rate. This is
what licenses transporting the map off the tested cells. It is weaker than
assuming the tested cells are a random sample of the grid -- which they are not,
being enriched for well-studied, broadly-tuned receptors -- because it conditions
on the score, but it is not assumption-free: a pair can be untested precisely
because nobody expected it to bind, in a way the score does not capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = [
    "fit_calibration_map",
    "reference_calibration_map",
    "density_fill",
    "FillResult",
    "receptor_weighted_prevalence",
    "bin_reweighted_prevalence",
]


def fit_calibration_map(scores, labels) -> IsotonicRegression:
    """Fit the score -> binding-frequency map on tested cells only.

    Fit once and reuse. Refitting per matrix would make the extant and ancestral
    binarisations answer to different maps, which is the thing this module exists
    to prevent.
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=float).ravel()
    if scores.shape != labels.shape:
        raise ValueError(
            f"scores and labels must be the same length, got {scores.shape} and {labels.shape}"
        )
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(scores, labels)
    return iso


_REF_MAP_CACHE: dict = {}


def reference_calibration_map(prob_path, pairs_path, fasta_path) -> IsotonicRegression:
    """The one calibration map, fitted on the reference-tree human tested cells.

    Both notebooks call this so neither has to refit its own. The extant notebook
    fills with `tested_mask` set and the measurements overlaid; the ancestral
    notebook passes this same map with `tested_mask=None`, applying steps 1-4 to
    every cell of the ASR grid with no overlay.

    Reproduces the frozen experimental layer exactly: exact amino-acid sequence
    match against the reference FASTA, human rows only, `Responsive` maxed over
    replicate measurements of the same (sequence, odorant) pair. Cached per path
    triple, so repeated calls in one session return the identical fitted object.
    """
    import pandas as pd

    key = (str(prob_path), str(pairs_path), str(fasta_path))
    if key in _REF_MAP_CACHE:
        return _REF_MAP_CACHE[key]

    prob = pd.read_csv(prob_path, index_col=0)
    hum_pids = [i for i in prob.index if str(i).startswith("9606")]
    P = prob.loc[hum_pids]
    lig_cols = list(P.columns)
    Pv = P.values

    pid2seq, pid, buf = {}, None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if pid:
                    pid2seq[pid] = "".join(buf).upper()
                pid, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if pid:
        pid2seq[pid] = "".join(buf).upper()

    seq2pid = {}
    for p in hum_pids:
        seq2pid.setdefault(pid2seq[p], []).append(p)

    m = pd.read_csv(
        pairs_path,
        usecols=["Species", "UniProt ID", "Sequence", "Responsive", "smiles_id", "mutation"],
    )
    hs = m[m["Species"].str.lower().str.contains("homo")].copy()
    hs["Sequence"] = hs["Sequence"].str.upper().str.strip()
    hs["pid"] = hs["Sequence"].map(lambda s: seq2pid.get(s, [None])[0])
    matched = hs[hs["pid"].notna() & hs["smiles_id"].isin(set(lig_cols))]
    exp = matched.groupby(["pid", "smiles_id"])["Responsive"].max().reset_index()

    ridx = {p: i for i, p in enumerate(hum_pids)}
    cidx = {c: j for j, c in enumerate(lig_cols)}
    rr = np.array([ridx[p] for p in exp["pid"]])
    cc = np.array([cidx[c] for c in exp["smiles_id"]])

    iso = fit_calibration_map(Pv[rr, cc], exp["Responsive"].values.astype(int))
    _REF_MAP_CACHE[key] = iso
    return iso


@dataclass
class FillResult:
    """Outcome of a density fill, with the numbers needed to report it."""

    matrix: np.ndarray                  # int8, same shape as the score grid
    cut: float                          # score of the Nth ranked cell -- an output
    density: float                      # fraction of the full grid set to 1
    n_expected: int                     # N, expected true binders among untested
    n_experimental_pos: int             # measured positives written over the fill
    n_untested: int
    orphan_receptors: int               # rows summing to 0
    orphan_ligands: int                 # columns summing to 0
    calibrated_mean: float              # mean calibrated p over untested cells
    notes: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"grid density        : {self.density:.4%}\n"
            f"density cut (output): {self.cut:.4f}\n"
            f"orphaned receptors  : {self.orphan_receptors}"
        )


def density_fill(
    P: np.ndarray,
    iso: IsotonicRegression,
    tested_mask: np.ndarray | None = None,
    experimental: np.ndarray | None = None,
) -> FillResult:
    """Binarise a score grid by calibrated density and rank.

    Parameters
    ----------
    P
        Score grid, receptors x ligands.
    iso
        Map from `fit_calibration_map`. Never refit inside this function.
    tested_mask
        Boolean, same shape as `P`, True where a measurement exists. Pass None
        (the ancestral case) to treat every cell as untested, which applies
        steps 1-4 to the whole grid with no overlay.
    experimental
        Measured labels aligned to `P`, read only where `tested_mask` is True.
        Required whenever `tested_mask` is given.

    Returns
    -------
    FillResult
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2:
        raise ValueError(f"P must be 2-D, got shape {P.shape}")

    if tested_mask is None:
        tested_mask = np.zeros(P.shape, dtype=bool)
    else:
        tested_mask = np.asarray(tested_mask, dtype=bool)
        if tested_mask.shape != P.shape:
            raise ValueError(
                f"tested_mask shape {tested_mask.shape} does not match P {P.shape}"
            )
        if experimental is None:
            raise ValueError("experimental labels are required when tested_mask is given")

    untested = ~tested_mask
    n_untested = int(untested.sum())
    if n_untested == 0:
        raise ValueError("every cell is marked tested; nothing to fill")

    # 1-2. calibrated binding probability for each untested cell
    scores_untested = P[untested]
    p_cal = iso.predict(scores_untested)

    # 3. expected number of true binders among them
    n_expected = int(round(float(p_cal.sum())))
    n_expected = max(0, min(n_expected, n_untested))

    # 4. rank by model score, top N on. The Nth score is the cut, an output.
    X = np.zeros(P.shape, dtype=np.int8)
    flat_positions = np.flatnonzero(untested.ravel())
    if n_expected > 0:
        order = np.argsort(scores_untested)[::-1]          # descending
        chosen = flat_positions[order[:n_expected]]
        X.ravel()[chosen] = 1
        cut = float(scores_untested[order[n_expected - 1]])
    else:
        cut = float("nan")

    # 5. measurement always wins
    n_exp_pos = 0
    if tested_mask.any():
        exp_vals = np.asarray(experimental)[tested_mask].astype(np.int8)
        X[tested_mask] = exp_vals
        n_exp_pos = int(exp_vals.sum())

    return FillResult(
        matrix=X,
        cut=cut,
        density=float(X.mean()),
        n_expected=n_expected,
        n_experimental_pos=n_exp_pos,
        n_untested=n_untested,
        orphan_receptors=int((X.sum(axis=1) == 0).sum()),
        orphan_ligands=int((X.sum(axis=0) == 0).sum()),
        calibrated_mean=float(p_cal.mean()),
    )


# --- independent prevalence checks -------------------------------------------
# Neither feeds the fill. Both exist so the density it lands on can be compared
# against estimates built a different way.

def receptor_weighted_prevalence(row_ids, labels) -> float:
    """Mean over receptors of each receptor's own tested binding rate.

    Unweights the testing effort: a receptor screened 300 times counts the same
    as one screened 5 times. Guards against the pair-weighted rate being pulled
    up by heavily-screened, broadly-tuned receptors.
    """
    row_ids = np.asarray(row_ids)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean([labels[row_ids == r].mean() for r in np.unique(row_ids)]))


def bin_reweighted_prevalence(
    tested_scores, tested_labels, grid_scores, n_bins: int = 20
) -> float:
    """Grid prevalence estimated by reweighting per-score-bin measured rates.

    Bins cells by model score, takes the measured binding rate inside each bin
    from the tested cells, and reweights those rates by how the *full grid*
    distributes over the bins. A coarse, non-parametric analogue of the isotonic
    map, so agreement between the two is a genuine check rather than a restatement.

    Bins are equal-width on [0, 1] -- the natural scale for a probability, and
    stable at 2.79-2.82% for any n_bins from 10 to 100 on this grid. Equal-count
    (quantile) bins are a poor choice here: the score distribution piles up near
    zero, so the top quantile bin spans most of the usable range and lumps strong
    and weak scores together, biasing the estimate upward (4.4% at 5 bins,
    decaying to 2.78% only by 100).
    """
    tested_scores = np.asarray(tested_scores, dtype=float).ravel()
    tested_labels = np.asarray(tested_labels, dtype=float).ravel()
    grid_scores = np.asarray(grid_scores, dtype=float).ravel()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    t_bin = np.digitize(tested_scores, edges[1:-1])
    g_bin = np.digitize(grid_scores, edges[1:-1])

    total = 0.0
    for b in range(n_bins):
        in_grid = float((g_bin == b).sum())
        if in_grid == 0:
            continue
        m = t_bin == b
        rate = tested_labels[m].mean() if m.any() else 0.0
        total += rate * in_grid
    return float(total / len(grid_scores))
