"""Robustness checks on c05.

  1  rebuild the matched control under several seeds (AUCs and functional-group odds ratios)
  2  re-score the AUCs with a Bemis-Murcko scaffold split
  3  compare the Q1 clustering with matched and random natural-product sets, and put a
     bootstrap interval on it

    python c06_robustness.py            # 5 seeds, 5 null sets
    python c06_robustness.py --quick    # 2 and 2

Outputs in results/: robust_matched_auc.csv, robust_scaffold_auc.csv, robust_fg_matched.csv,
robust_knn_null.csv, robust_enrichment_ci.csv, robustness_report.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import c05_analyze as analyze
import chemspace as cs

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

NEIGHBOURS = analyze.NEIGHBOURS
DESCRIPTORS = cs.DESCRIPTOR_NAMES

# Resamples for the bootstrap interval on the kNN enrichment. 2000 is enough for a
# stable 95% percentile interval and costs nothing, because it resamples the per-odorant
# purities that are already computed rather than redoing any neighbour search.
BOOTSTRAP_RESAMPLES = 2000

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


# --------------------------------------------------------------- scaffold splitting
def murcko_scaffolds(smiles_list):
    """Bemis-Murcko scaffold of each molecule, as a SMILES string.

    The scaffold is what is left after stripping every side chain: the ring systems and
    the linkers between them. Two molecules with the same scaffold are the same chemotype
    decorated differently, which is exactly the kind of pair that must not straddle a
    train/test split.

    A molecule with no rings -- most short-chain odorants -- has an empty scaffold. Those
    are given their own singleton group rather than being pooled into one enormous
    "acyclic" group, because pooling them would force every acyclic molecule into the
    same fold and wreck the class balance.

    Returns
    -------
    ndarray of str
        One group label per molecule.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = []
    for position, smiles in enumerate(smiles_list):
        label = f"acyclic_{position}"
        molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if molecule is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
                if scaffold:
                    label = scaffold
            except Exception:
                pass
        groups.append(label)
    return np.asarray(groups)


def auc_scaffold_split(features, labels, groups, seed=0, folds=analyze.FOLDS):
    """ROC AUC where no scaffold appears in both train and test.

    `StratifiedGroupKFold` keeps each scaffold whole while holding the odorant/partner
    balance roughly even across folds. Compare the result against
    `analyze.auc_cross_validated` on the same data: the gap between them is how much of the
    random-CV number was memorised chemotype rather than learned chemistry.
    """
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    predicted = cross_val_predict(
        model, features, labels, cv=splitter, groups=groups, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(labels, predicted))


# ------------------------------------------------------------------- the three parts
def matched_across_seeds(universe, normalised, odorant_rows, background_rows,
                         volatility, seeds):
    """Attack 1 and 2 -- rebuild the matched control under each seed, both ways.

    For every seed the matching is redone from scratch, balance is rechecked, and each
    feature set is scored twice: once with random cross-validation (comparable to c05)
    and once with a scaffold split (the honest number).

    Returns
    -------
    random_cv, scaffold : DataFrame
        One row per (seed, feature set).
    """
    log("\n" + "=" * 72)
    log("ATTACKS 1 & 2  DOES THE MATCHED CONTROL HOLD UP?")
    log("=" * 72)
    log(f"  rebuilding the control under {len(seeds)} matching seeds, scoring each")
    log("  with random CV and with a Bemis-Murcko scaffold split")

    descriptor_values = universe[DESCRIPTORS].values.astype(float)
    feature_sets = {
        "volatility_only": volatility,
        "descriptors_22": descriptor_values,
        "molformer_768": normalised,
    }

    random_rows, scaffold_rows, group_rows = [], [], []
    for seed in seeds:
        matched_odorants, matched_background, fraction = cs.match_on(
            universe, analyze.MATCH_COVARIATES, odorant_rows, background_rows,
            caliper_sd=analyze.CALIPER_SD, seed=seed,
        )
        pairs = np.r_[matched_odorants, matched_background]
        labels = np.r_[np.ones(len(matched_odorants), int),
                       np.zeros(len(matched_background), int)]

        worst = max(
            abs(cs.smd(universe[covariate].values.astype(float)[matched_odorants],
                       universe[covariate].values.astype(float)[matched_background]))
            for covariate in analyze.MATCH_COVARIATES
        )
        groups = murcko_scaffolds(universe["smiles"].values[pairs])

        log(f"\n  seed {seed}: matched {len(matched_odorants)} ({fraction:.1%}), "
            f"worst |SMD| {worst:.4f} "
            f"({'PASS' if worst < analyze.BALANCE_TOLERANCE else 'FAIL'}), "
            f"{len(set(groups))} distinct scaffolds")

        for name, features in feature_sets.items():
            random_auc = analyze.auc_cross_validated(features[pairs], labels, seed=seed)
            scaffold_auc = auc_scaffold_split(features[pairs], labels, groups, seed=seed)
            log(f"    {name:18s} random CV {random_auc:.3f}   "
                f"scaffold {scaffold_auc:.3f}   "
                f"optimism {random_auc - scaffold_auc:+.3f}")

            common = {"seed": seed, "features": name, "n_pairs": len(matched_odorants),
                      "fraction_matched": fraction, "worst_smd": worst}
            random_rows.append({**common, "auc": random_auc, "split": "random"})
            scaffold_rows.append({**common, "auc": scaffold_auc, "split": "scaffold",
                                  "n_scaffolds": len(set(groups))})

        # The matched functional-group odds ratios are the reported ones, so they have
        # to be shown not to depend on which partner each odorant happened to get.
        for group in cs.FUNCTIONAL_GROUP_NAMES:
            present = universe[group].values > 0
            paired = cs.paired_odds_ratio(present[matched_odorants],
                                          present[matched_background])
            group_rows.append({"seed": seed, "group": group,
                               "odds_ratio": paired["odds_ratio"],
                               "ci_lo": paired["ci_lo"], "ci_hi": paired["ci_hi"],
                               "p": paired["p"]})

    random_cv = pd.DataFrame(random_rows)
    scaffold = pd.DataFrame(scaffold_rows)
    by_group = pd.DataFrame(group_rows)
    random_cv.to_csv(RESULTS / "robust_matched_auc.csv", index=False)
    scaffold.to_csv(RESULTS / "robust_scaffold_auc.csv", index=False)
    by_group.to_csv(RESULTS / "robust_fg_matched.csv", index=False)

    log("\n  across seeds (mean +/- sd):")
    log(f"    {'features':18s} {'random CV':>16s} {'scaffold':>16s}")
    for name in feature_sets:
        r = random_cv[random_cv.features == name]["auc"]
        s = scaffold[scaffold.features == name]["auc"]
        log(f"    {name:18s} {r.mean():8.3f} +/-{r.std():5.3f} "
            f"{s.mean():8.3f} +/-{s.std():5.3f}")

    log("\n  Seed-to-seed spread is the matcher's contribution; the random-to-scaffold")
    log("  gap is how much of the AUC was near-duplicate chemistry. Quote the")
    log("  scaffold column.")

    log("\n  matched functional-group odds ratios across seeds:")
    log(f"    {'group':22s} {'mean':>7s} {'sd':>6s} {'min':>7s} {'max':>7s}")
    summary = by_group.groupby("group")["odds_ratio"].agg(["mean", "std", "min", "max"])
    for group, row in summary.sort_values("mean", ascending=False).iterrows():
        log(f"    {group:22s} {row['mean']:7.2f} {row['std']:6.3f} "
            f"{row['min']:7.2f} {row['max']:7.2f}")
    return random_cv, scaffold


def knn_enrichment(normalised, query_rows, member_mask, universe_size):
    """Enrichment of a set among its own members' nearest neighbours.

    The same quantity c05's Q1 computes, written so it can be pointed at any set:
    the share of a query's `NEIGHBOURS` nearest neighbours that also belong to the set,
    divided by the share of the whole universe the set occupies.

    Returns
    -------
    purity : ndarray
        Per-query neighbour share, kept so it can be bootstrapped.
    enrichment, base_rate : float
    """
    neighbours, _ = cs.knn(normalised, query_rows, NEIGHBOURS)
    purity = member_mask[neighbours].mean(1)
    base_rate = member_mask.sum() / universe_size
    return purity, float(purity.mean() / base_rate), float(base_rate)


def enrichment_nulls(universe, normalised, odorant_rows, background_rows,
                     is_odorant, null_sets, seed=0):
    """Attack 3 -- Q1's enrichment against a matched null and a random null."""
    log("\n" + "=" * 72)
    log("ATTACK 3  IS THE kNN ENRICHMENT BIG, OR ONLY BIG-LOOKING?")
    log("=" * 72)

    universe_size = len(universe)
    rows = []

    # The observed value, recomputed here so the report stands alone.
    purity, enrichment, base_rate = knn_enrichment(
        normalised, odorant_rows, is_odorant, universe_size
    )
    rows.append({"set": "odorants", "replicate": 0, "n": len(odorant_rows),
                 "purity": float(purity.mean()), "base_rate": base_rate,
                 "enrichment": enrichment})
    log(f"\n  odorants                  : {enrichment:6.1f}x  "
        f"(purity {purity.mean():.4f}, base rate {base_rate:.4f})")

    # The null that matters: a set matched to the odorants on transport properties.
    log("\n  matched natural products (the null that matters):")
    matched_enrichments = []
    for replicate in range(null_sets):
        _, matched_background, _ = cs.match_on(
            universe, analyze.MATCH_COVARIATES, odorant_rows, background_rows,
            caliper_sd=analyze.CALIPER_SD, seed=seed + replicate,
        )
        mask = np.zeros(universe_size, bool)
        mask[matched_background] = True
        _, value, rate = knn_enrichment(normalised, matched_background, mask,
                                        universe_size)
        matched_enrichments.append(value)
        rows.append({"set": "matched_np", "replicate": replicate,
                     "n": len(matched_background), "purity": np.nan,
                     "base_rate": rate, "enrichment": value})
        log(f"    seed {seed + replicate}: {value:6.1f}x")

    # The sanity check: any random set of the same size should show no clustering.
    log("\n  random natural products (sanity check, should be ~1x):")
    rng = np.random.default_rng(seed)
    random_enrichments = []
    for replicate in range(null_sets):
        draw = rng.choice(background_rows, len(odorant_rows), replace=False)
        mask = np.zeros(universe_size, bool)
        mask[draw] = True
        _, value, rate = knn_enrichment(normalised, draw, mask, universe_size)
        random_enrichments.append(value)
        rows.append({"set": "random_np", "replicate": replicate, "n": len(draw),
                     "purity": np.nan, "base_rate": rate, "enrichment": value})
        log(f"    draw {replicate}: {value:6.1f}x")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "robust_knn_null.csv", index=False)

    matched_mean = float(np.mean(matched_enrichments))
    random_mean = float(np.mean(random_enrichments))
    log("\n  " + "-" * 68)
    log(f"    observed (odorants)      {enrichment:7.1f}x")
    log(f"    matched null             {matched_mean:7.1f}x  "
        f"+/- {np.std(matched_enrichments):.1f}")
    log(f"    random null              {random_mean:7.1f}x  "
        f"+/- {np.std(random_enrichments):.1f}")
    log(f"    observed / matched null  {enrichment / matched_mean:7.2f}x  "
        "<-- THE NUMBER TO QUOTE")
    log("\n  The raw enrichment counts the transport effect as if it were an olfactory")
    log("  one. Against the matched null it is a factor of a few, not a factor of 47.")
    return table, purity, enrichment


def bootstrap_interval(purity, base_rate, resamples=BOOTSTRAP_RESAMPLES, seed=0):
    """Percentile bootstrap interval on the kNN enrichment.

    Resamples odorants, not neighbours: the uncertainty being described is "would
    another draw of odorants from the same population give the same enrichment", which
    is the question a reader asks. Cheap, because the neighbour search is already done.
    """
    log("\n" + "=" * 72)
    log("INTERVAL  HOW PRECISE IS THE ENRICHMENT?")
    log("=" * 72)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(purity), size=(resamples, len(purity)))
    values = purity[draws].mean(axis=1) / base_rate

    low, high = np.percentile(values, [2.5, 97.5])
    table = pd.DataFrame([{
        "statistic": "knn_enrichment",
        "estimate": float(purity.mean() / base_rate),
        "ci_lo": float(low), "ci_hi": float(high),
        "bootstrap_sd": float(values.std()),
        "n_odorants": len(purity), "resamples": resamples,
    }])
    table.to_csv(RESULTS / "robust_enrichment_ci.csv", index=False)

    log(f"  enrichment {purity.mean() / base_rate:.1f}x   "
        f"95% CI [{low:.1f}, {high:.1f}]   ({resamples} resamples)")
    log("  The interval is narrow because n is large. Narrow is not the same as")
    log("  correct: it describes sampling noise only, never the matched null above.")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5,
                        help="matching seeds for attacks 1 and 2 (default 5)")
    parser.add_argument("--null-sets", type=int, default=5,
                        help="replicates per null in attack 3 (default 5)")
    parser.add_argument("--quick", action="store_true",
                        help="2 seeds and 2 null sets, for a smoke test")
    arguments = parser.parse_args()

    n_seeds = 2 if arguments.quick else arguments.seeds
    n_nulls = 2 if arguments.quick else arguments.null_sets

    universe, _, normalised = analyze.load_universe()
    is_odorant = universe.is_odorant.values == 1
    odorant_rows = np.flatnonzero(is_odorant)
    background_rows = np.flatnonzero(universe.in_background.values == 1)

    volatility = universe[["Tb_joback"]].values.astype(float)
    volatility = np.where(np.isfinite(volatility), volatility,
                          np.nanmedian(volatility))

    log(f"universe {len(universe)}  odorants {len(odorant_rows)}  "
        f"background {len(background_rows)}")
    log(f"seeds {n_seeds}  null replicates {n_nulls}")

    matched_across_seeds(universe, normalised, odorant_rows, background_rows,
                         volatility, list(range(n_seeds)))
    _, purity, _ = enrichment_nulls(universe, normalised, odorant_rows,
                                    background_rows, is_odorant, n_nulls)
    bootstrap_interval(purity, is_odorant.sum() / len(universe))

    log("\n" + "=" * 72)
    log("WHAT TO PUT IN THE PAPER")
    log("=" * 72)
    log("  * the scaffold-split AUCs, not the random-CV ones")
    log("  * the kNN enrichment beside its matched null, never on its own")
    log("  * the matched functional-group odds ratios from c05's fg_enrichment.csv,")
    log("    with their seed spread from robust_fg_matched.csv")
    log("  * the seed spread of the AUCs, as evidence the control is not one lucky pairing")

    (RESULTS / "robustness_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'robustness_report.txt'}")


if __name__ == "__main__":
    main()
