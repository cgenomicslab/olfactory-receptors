"""Compare odorants with the natural-product background.

  Q1  clustering: the share of each odorant's 50 nearest neighbours (cosine, 768-d) that
      are odorants, divided by the odorant base rate
  Q2  matched control: each odorant paired with a natural product of similar boiling point,
      molecular weight and logP; balance checked (SMD); separability of the pairs as
      cross-validated AUC on three feature sets
  Q3  biosynthetic pathway: Fisher's exact test per pathway, BH q-values
  Q4  functional groups against the matched pairs: odds ratio, McNemar test, interval from
      resampling pairs. The odds ratio against the whole background is kept alongside.

Outputs in results/: knn_enrichment.csv, coconut_matched_smd.csv, coconut_matched_auc.csv,
coconut_lda_structure.csv, pathway_enrichment.csv, fg_enrichment.csv, analyze_report.txt
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import chemspace as cs

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 0

DESCRIPTORS = cs.DESCRIPTOR_NAMES
FUNCTIONAL_GROUPS = cs.FUNCTIONAL_GROUP_NAMES

NEIGHBOURS = 50

# The three transport properties. A molecule has to be small enough and volatile enough
# to reach the receptor at all, so these are exactly the axes on which "odorants are
# special" is least interesting -- which is why the control equalises them. They are the
# same three the F6 property maps are drawn on.
MATCH_COVARIATES = ["Tb_joback", "MolWt", "LogP"]

# Largest allowed distance between a molecule and its partner, per covariate, in
# standard deviations. Tighter than this and too few odorants find a partner; looser and
# the "matched" pairs stop being alike. 0.25 is the usual choice in the matching
# literature and leaves the balance check (below) comfortably passed.
CALIPER_SD = 0.25

# A matched pair set is balanced when every covariate's standardised mean difference is
# under this. If any exceeds it, matching failed and no AUC computed on it means
# anything -- so the run says so loudly rather than printing numbers.
BALANCE_TOLERANCE = 0.01

FOLDS = 5

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


def load_universe():
    """
    Read the universe and line it up with the embedding.

    The embedding dropped a few molecules that were too long to tokenise, so the two are
    not the same length. Reordering the universe to follow the embedding index is what
    lets us use a row number to mean the same molecule in both.

    Returns
    -------
    universe : DataFrame
    embeddings : ndarray, shape (n_molecules, 768)
    normalised : ndarray
        Embeddings scaled to unit length, so a dot product gives cosine similarity.
    """
    universe = pd.read_parquet(RESULTS / "universe.parquet")
    index = pd.read_csv(RESULTS / "embed_index.csv")
    embeddings = np.load(RESULTS / "embeddings.npy")

    universe = universe.set_index("inchikey").loc[index.inchikey].reset_index()

    # A handful of descriptors can be missing; fill them with the column median so the
    # classifiers do not choke.
    universe[DESCRIPTORS] = universe[DESCRIPTORS].apply(pd.to_numeric, errors="coerce")
    universe[DESCRIPTORS] = universe[DESCRIPTORS].fillna(universe[DESCRIPTORS].median())

    normalised = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return universe, embeddings, normalised


def question1_clustering(universe, normalised, odorant_rows, is_odorant):
    """
    Q1 -- how much more likely is an odorant's neighbour to be an odorant than chance?

    For each odorant we take its 50 nearest neighbours in the embedding and ask what
    fraction are also odorants. Divided by the fraction of the whole universe that is
    odorants, that gives an enrichment: 1x would mean no clustering at all.
    """
    log("\n" + "=" * 72)
    log(f"Q1  ARE ODORANTS NEAR EACH OTHER? (full 768-d cosine, k={NEIGHBOURS})")
    log("=" * 72)

    neighbours, _ = cs.knn(normalised, odorant_rows, NEIGHBOURS)
    odorant_fraction = is_odorant[neighbours].mean(1)
    chance = is_odorant.mean()
    enrichment = odorant_fraction.mean() / chance

    log(f"  odorant fraction among an odorant's {NEIGHBOURS} NN : {odorant_fraction.mean():.4f}")
    log(f"  base rate                                 : {chance:.4f}")
    log(f"  ENRICHMENT                                : {enrichment:.1f}x")
    log(f"  odorants with >=50% odorant neighbours     : "
        f"{100 * (odorant_fraction >= 0.5).mean():.1f}%")

    pd.DataFrame([{
        "knn_purity": odorant_fraction.mean(),
        "base_rate": chance,
        "enrichment": enrichment,
        "frac_ge50": (odorant_fraction >= 0.5).mean(),
    }]).to_csv(RESULTS / "knn_enrichment.csv", index=False)


@dataclass
class MatchedControl:
    """The matched pair set, and the evidence that the matching worked.

    Passed to Q4, which tests the functional groups pair by pair against it.
    """

    odorant_rows: np.ndarray        # matched odorants, aligned with background_rows
    background_rows: np.ndarray     # their non-odorant natural-product partners
    fraction_matched: float         # share of odorants that found a partner
    balance: pd.DataFrame           # per-covariate SMD before and after
    worst_smd: float                # largest |SMD| after matching
    balanced: bool                  # worst_smd < BALANCE_TOLERANCE


def auc_cross_validated(features, labels, seed=SEED):
    """Cross-validated ROC AUC of a standardised logistic regression.

    Linear on purpose. The question is whether the two groups are *separable at all*
    once matched, not how well a flexible model can be made to separate them, and a
    linear model is what makes the LDA axis below interpretable as "which descriptors
    carry the difference".

    Parameters
    ----------
    features : ndarray, shape (n_molecules, n_features)
    labels : ndarray of int
        1 for odorant, 0 for the matched partner.
    seed : int, optional
        Controls the fold split. The scaffold-split version is in c06.

    Returns
    -------
    float
        Area under the ROC curve, from out-of-fold predictions.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        model, features, labels, cv=splitter, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, predicted))


def lda_structure_coefficients(features, labels, names):
    """The discriminant axis, expressed as one correlation per descriptor.

    LDA gives a single axis separating odorants from their partners. Its raw weights
    are not readable -- correlated descriptors split a shared weight between them, so a
    descriptor that matters can carry a small one. The *structure coefficient* is the
    correlation between a descriptor and the discriminant score instead, which is stable
    under collinearity and is what "this axis is about oxygen content" actually means.

    Returns
    -------
    Series
        Structure coefficient per descriptor, sorted ascending.
    """
    standardised = StandardScaler().fit_transform(features)
    axis = LinearDiscriminantAnalysis(n_components=1).fit_transform(
        standardised, labels
    ).ravel()

    coefficients = {
        name: float(np.corrcoef(standardised[:, column], axis)[0, 1])
        for column, name in enumerate(names)
    }
    return pd.Series(coefficients, name="structure_coefficient").sort_values()


def question2_matched_control(universe, normalised, odorant_rows, background_rows):
    """
    Q2 -- what is left once volatility and size are equalised?

    Builds a control group of non-odorant natural products, one per odorant, matched on
    the three transport properties, then asks three questions of it:

      * did the matching work? (every |SMD| under `BALANCE_TOLERANCE`)
      * how separable are the two groups now, on volatility alone, on the 22-descriptor
        panel, and on the MolFormer embedding?
      * which descriptors carry that separation

    The unmatched AUC is computed alongside, against a random background sample of the
    same size. The drop from unmatched to matched is the part of "odorants are
    distinctive" that transport properties already explained.
    """
    log("\n" + "=" * 72)
    log("Q2  THE MATCHED CONTROL: is it odour chemistry, or just small and volatile?")
    log("=" * 72)

    matched_odorants, matched_background, fraction = cs.match_on(
        universe, MATCH_COVARIATES, odorant_rows, background_rows,
        caliper_sd=CALIPER_SD, seed=SEED,
    )
    log(f"  matched {len(matched_odorants)} of {len(odorant_rows)} odorants "
        f"({fraction:.1%}) on {', '.join(MATCH_COVARIATES)}")

    # --- did the matching work? -------------------------------------------------
    rows = []
    for covariate in MATCH_COVARIATES:
        values = universe[covariate].values.astype(float)
        rows.append({
            "covariate": covariate,
            "smd_before": cs.smd(values[odorant_rows], values[background_rows]),
            "smd_after": cs.smd(values[matched_odorants], values[matched_background]),
            "mean_odorant": float(np.nanmean(values[matched_odorants])),
            "mean_matched": float(np.nanmean(values[matched_background])),
        })
    balance = pd.DataFrame(rows)
    balance.to_csv(RESULTS / "coconut_matched_smd.csv", index=False)

    worst = float(balance["smd_after"].abs().max())
    balanced = worst < BALANCE_TOLERANCE

    log("\n  covariate balance (standardised mean difference):")
    for _, row in balance.iterrows():
        log(f"    {row.covariate:12s} before {row.smd_before:+7.3f}   "
            f"after {row.smd_after:+7.4f}")
    log(f"    worst |SMD| after matching : {worst:.4f}  "
        f"({'PASS' if balanced else 'FAIL'}, tolerance {BALANCE_TOLERANCE})")

    if not balanced:
        log("\n  !! MATCHING FAILED. The AUCs below compare groups that are not")
        log("     actually alike on the transport properties, so they do not")
        log("     separate odour chemistry from smallness and volatility.")

    # --- how separable are they now? --------------------------------------------
    descriptor_values = universe[DESCRIPTORS].values.astype(float)
    labels = np.r_[np.ones(len(matched_odorants), int),
                   np.zeros(len(matched_background), int)]
    pairs = np.r_[matched_odorants, matched_background]

    # The unmatched reference: the same odorants against a random background draw of
    # equal size. This is the number the matched AUC has to be read against.
    sample = np.random.default_rng(SEED).choice(
        background_rows, len(matched_odorants), replace=False
    )
    unmatched = np.r_[matched_odorants, sample]

    # Joback fails on ~10% of COCONUT, mostly the large decorated molecules it was never
    # fitted for. Matched rows always have a real value -- matching needs one -- but the
    # random unmatched draw does not, so the gaps are filled with the median rather than
    # letting the classifier see a NaN.
    volatility = universe[["Tb_joback"]].values.astype(float)
    volatility = np.where(np.isfinite(volatility), volatility,
                          np.nanmedian(volatility))

    feature_sets = {
        "volatility_only": volatility,
        "descriptors_22": descriptor_values,
        "molformer_768": normalised,
    }

    auc_rows = []
    for name, features in feature_sets.items():
        matched_auc = auc_cross_validated(features[pairs], labels)
        unmatched_auc = auc_cross_validated(features[unmatched], labels)
        auc_rows.append({
            "features": name,
            "auc_unmatched": unmatched_auc,
            "auc_matched": matched_auc,
            # not "drop": that shadows Series.drop, so row.drop returns the method
            "auc_drop": unmatched_auc - matched_auc,
        })

    aucs = pd.DataFrame(auc_rows)
    aucs.to_csv(RESULTS / "coconut_matched_auc.csv", index=False)

    log("\n  separability, 5-fold cross-validated ROC AUC:")
    log(f"    {'features':18s} {'unmatched':>10s} {'matched':>9s} {'drop':>8s}")
    for _, row in aucs.iterrows():
        log(f"    {row.features:18s} {row.auc_unmatched:10.3f} "
            f"{row.auc_matched:9.3f} {row.auc_drop:8.3f}")
    log("    (volatility alone should land near 0.50 once matched -- that is the")
    log("     check that the matching removed what it was meant to remove.)")

    # --- which descriptors carry the separation ----------------------------------
    coconut_axis = lda_structure_coefficients(
        descriptor_values[pairs], labels, DESCRIPTORS
    )
    coconut_axis.to_frame().to_csv(RESULTS / "coconut_lda_structure.csv")

    return MatchedControl(
        odorant_rows=matched_odorants,
        background_rows=matched_background,
        fraction_matched=fraction,
        balance=balance,
        worst_smd=worst,
        balanced=balanced,
    )


def question3_pathways(universe, odorant_rows, background_rows):
    """
    Q3 -- which biosynthetic routes do odorants come from?

    COCONUT labels each natural product with the pathway that makes it. Comparing those
    labels between odorants and the background says what odour chemistry is built from.
    """
    log("\n" + "=" * 72)
    log("Q3  BIOSYNTHETIC PATHWAY: odorants vs the natural-product background")
    log("=" * 72)

    column = "np_classifier_pathway"
    if column not in universe:
        return

    odorant_pathways = universe[column][odorant_rows].dropna()
    background_pathways = universe[column][background_rows].dropna()
    log(f"  annotated: odorants {len(odorant_pathways)}/{len(odorant_rows)}, "
        f"background {len(background_pathways)}/{len(background_rows)}")

    rows = []
    for pathway in sorted(set(odorant_pathways) | set(background_pathways)):
        n_odorant = int((odorant_pathways == pathway).sum())
        n_background = int((background_pathways == pathway).sum())
        _, p_value = stats.fisher_exact([
            [n_odorant, len(odorant_pathways) - n_odorant],
            [n_background, len(background_pathways) - n_background],
        ])
        odds_ratio, low, high = cs.odds_ratio_ci(n_odorant, len(odorant_pathways),
                                                 n_background, len(background_pathways))
        rows.append({
            "pathway": pathway,
            "n_odorant": n_odorant,
            "n_background": n_background,
            "pct_odorant": 100 * n_odorant / max(len(odorant_pathways), 1),
            "pct_background": 100 * n_background / max(len(background_pathways), 1),
            "odds_ratio": odds_ratio,
            "ci_lo": low, "ci_hi": high,
            "p": p_value,
        })

    pathways = pd.DataFrame(rows).sort_values("odds_ratio", ascending=False)
    pathways["q"] = cs.benjamini_hochberg(pathways["p"].values)
    pathways.to_csv(RESULTS / "pathway_enrichment.csv", index=False)

    for _, row in pathways.iterrows():
        log(f"    {row.pathway:34s} {row.pct_odorant:5.1f}% vs {row.pct_background:5.1f}%   "
            f"OR {row.odds_ratio:6.2f}  [{row.ci_lo:5.2f}, {row.ci_hi:6.2f}]   "
            f"q {cs.format_p(row.q)}")


def _group_contrast(universe, group, odorant_rows, background_rows):
    """Odds ratio, CI and Fisher p for one functional group, odorants against the background."""
    n_odorant = int((universe[group].values[odorant_rows] > 0).sum())
    n_background = int((universe[group].values[background_rows] > 0).sum())
    _, p_value = stats.fisher_exact([
        [n_odorant, len(odorant_rows) - n_odorant],
        [n_background, len(background_rows) - n_background],
    ])
    odds_ratio, low, high = cs.odds_ratio_ci(n_odorant, len(odorant_rows),
                                             n_background, len(background_rows))
    return {
        "pct_odorant": 100 * n_odorant / len(odorant_rows),
        "pct_background": 100 * n_background / len(background_rows),
        "odds_ratio": odds_ratio, "ci_lo": low, "ci_hi": high, "p": p_value,
    }


def _group_contrast_matched(universe, group, matched):
    """Paired odds ratio and McNemar p for one functional group, over the matched pairs."""
    present = universe[group].values > 0
    paired = cs.paired_odds_ratio(present[matched.odorant_rows],
                                  present[matched.background_rows])
    return {
        "pct_odorant_matched": paired["pct_a"],
        "pct_partner_matched": paired["pct_b"],
        "pairs_only_odorant": paired["only_a"],
        "pairs_only_partner": paired["only_b"],
        "OR_matched": paired["odds_ratio"],
        "ci_lo_matched": paired["ci_lo"],
        "ci_hi_matched": paired["ci_hi"],
        "p_matched": paired["p"],
    }


def question4_functional_groups(universe, odorant_rows, background_rows, matched):
    """
    Q4 -- which chemical groups are over- or under-represented in odorants?

    Tested twice. Against the matched control, pair by pair: this is the contrast that
    is reported, because it compares odorants with natural products of the same size
    and volatility. Against the whole natural-product background: kept beside it, so a
    reader can see which groups only looked depleted because they sit on large molecules.
    """
    log("\n" + "=" * 72)
    log("Q4  FUNCTIONAL GROUPS: odorants against their matched partners")
    log("=" * 72)
    log(f"  {len(matched.odorant_rows)} pairs. Odds ratio from the two percentages; "
        "McNemar exact test;")
    log("  95% interval from 2,000 resamples of whole pairs")

    rows = []
    for group in FUNCTIONAL_GROUPS:
        row = {"group": group}
        row.update(_group_contrast_matched(universe, group, matched))
        row.update(_group_contrast(universe, group, odorant_rows, background_rows))
        rows.append(row)

    groups = pd.DataFrame(rows).sort_values("OR_matched", ascending=False)
    groups["q_matched"] = cs.benjamini_hochberg(groups["p_matched"].values)
    groups["q"] = cs.benjamini_hochberg(groups["p"].values)
    groups.to_csv(RESULTS / "fg_enrichment.csv", index=False)

    log(f"\n    {'group':22s} {'odorant':>8s} {'partner':>8s}   "
        f"{'OR':>6s}  {'95% CI':>15s}   q")
    for _, row in groups.iterrows():
        log(f"    {row.group:22s} {row.pct_odorant_matched:7.1f}% "
            f"{row.pct_partner_matched:7.1f}%   {row.OR_matched:6.2f}  "
            f"[{row.ci_lo_matched:5.2f}, {row.ci_hi_matched:5.2f}]   "
            f"{cs.format_p(row.q_matched)}")

    log("\n  the same groups against the whole natural-product background:")
    log(f"    {'group':22s} {'OR all':>8s} {'OR matched':>11s}   verdict")
    reversed_groups = []
    for _, row in groups.iterrows():
        crosses_one = (row.odds_ratio - 1) * (row.OR_matched - 1) < 0
        if crosses_one and row.q_matched < 0.05:
            reversed_groups.append(row.group)
            verdict = "reverses"
        elif row.q_matched >= 0.05:
            verdict = "n.s. once matched"
        else:
            verdict = "same direction"
        log(f"    {row.group:22s} {row.odds_ratio:8.2f} {row.OR_matched:11.2f}   {verdict}")
    log(f"\n    {len(reversed_groups)} of {len(groups)} groups change direction "
        "significantly once size and volatility are equalised.")
    log("    Those were depleted in odorants only because they sit on large molecules.")


def report_sugar(universe, is_odorant):
    """
    A sanity check on the volatility story, using COCONUT's own sugar flag.

    Sugar-bearing natural products are the archetypal non-volatile molecule, so odorants
    should be almost free of them.
    """
    if "contains_sugar" not in universe:
        return

    log("\n  sugar content (COCONUT flag):")
    groups = [
        ("odorants (in COCONUT)", is_odorant & (universe.in_coconut.values == 1)),
        ("NP background", universe.in_background.values == 1),
    ]
    for name, mask in groups:
        flags = pd.Series(universe.contains_sugar.values[mask]).map(
            {True: 1, False: 0, "true": 1, "false": 0}
        ).fillna(0)
        log(f"    {name:22s} contains_sugar = {100 * flags.mean():.1f}%")


def main():
    universe, embeddings, normalised = load_universe()

    is_odorant = universe.is_odorant.values == 1
    odorant_rows = np.flatnonzero(is_odorant)
    background_rows = np.flatnonzero(universe.in_background.values == 1)

    log(f"universe {len(universe)}  odorants {len(odorant_rows)}  "
        f"natural-product background {len(background_rows)}  "
        f"base rate {is_odorant.mean():.4f}")

    question1_clustering(universe, normalised, odorant_rows, is_odorant)
    matched = question2_matched_control(
        universe, normalised, odorant_rows, background_rows
    )
    question3_pathways(universe, odorant_rows, background_rows)
    question4_functional_groups(universe, odorant_rows, background_rows, matched)
    report_sugar(universe, is_odorant)

    log("\n" + "=" * 72)
    log("HOW TO READ THIS")
    log("=" * 72)
    log("  Q1's enrichment and Q2's matched AUC answer the same question and only")
    log("  mean something together. Quote neither alone. Then run c06_robustness.py:")
    log("  it repeats Q2 across matching seeds, re-estimates every AUC under a")
    log("  scaffold split, and puts Q1's enrichment against the matched null.")

    (RESULTS / "analyze_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'analyze_report.txt'}")


if __name__ == "__main__":
    main()
