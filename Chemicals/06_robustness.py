"""Step 6 -- is any of this reportable?

`04` produces numbers. This produces the reasons to believe them, by attacking the
three places where the analysis could be fooling itself. Run it before quoting anything.

The three attacks
-----------------

**1. Did the matched control depend on one lucky pairing?**
`greedy_match` walks the odorants in a random order and gives each the nearest unused
partner, so a different order gives a different control group. If the matched AUC moves
much between seeds, it is a property of the matcher rather than of the chemistry. Here
the whole of Q2 is rebuilt from scratch under several seeds and the spread is reported.

**2. Are the AUCs inflated by near-duplicate molecules?**
Random cross-validation lets two molecules that differ by one methyl group land on
opposite sides of the train/test split, and a classifier that has memorised one will
"predict" the other. Chemical datasets are full of such series, so a random-CV AUC
systematically overstates how well a model generalises to genuinely new chemistry. The
fix is to split on Bemis-Murcko scaffold: every molecule sharing a core stays on the
same side, so the test fold is chemistry the model has not seen. The scaffold AUC is
the honest one, and it is the one to quote.

**3. Is the kNN enrichment large, or merely large-looking?**
Q1 says an odorant's neighbours are ~47x more likely to be odorants than chance. That
sounds decisive, but it has no scale until you know what a *non-answer* looks like. Two
nulls give it one:

    matched natural products   the Q2 control group, which is odorant-like in size and
                               volatility but is not odorant chemistry. If a set chosen
                               on transport properties alone already clusters at ~19x,
                               then most of the 47x is "small volatile molecules look
                               alike to MolFormer", not "odour chemistry is a region".
    random natural products    same size, drawn at random. Should land at ~1x. This is
                               the sanity check that the measurement works at all.

The number to quote is not 47x. It is 47x *against a matched null of ~19x* -- a ratio
of about 2.5. Reporting the 47x alone would be claiming the transport effect as an
olfactory one.

Outputs
-------
results/robust_matched_auc.csv     Q2 repeated across matching seeds
results/robust_scaffold_auc.csv    the same AUCs under a scaffold split
results/robust_knn_null.csv        Q1's enrichment against both nulls
results/robust_enrichment_ci.csv   bootstrap interval on Q1's enrichment
results/robustness_report.txt      everything printed below

Usage
-----
    python 06_robustness.py                  # 5 seeds, 5 random null sets
    python 06_robustness.py --seeds 10       # tighter interval, slower
    python 06_robustness.py --quick          # 2 seeds, 2 nulls, for a smoke test

A GPU makes it faster but is not required; on CPU the default run is roughly 10 minutes.
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

import importlib

import chemspace as cs

# Python cannot `import 04_analyze` -- a module name may not begin with a digit --
# so step 04 is loaded by name instead. Same module, same functions.
analyze = importlib.import_module("04_analyze")

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
    feature set is scored twice: once with random cross-validation (comparable to `04`)
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

    random_rows, scaffold_rows = [], []
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

    random_cv = pd.DataFrame(random_rows)
    scaffold = pd.DataFrame(scaffold_rows)
    random_cv.to_csv(RESULTS / "robust_matched_auc.csv", index=False)
    scaffold.to_csv(RESULTS / "robust_scaffold_auc.csv", index=False)

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
    return random_cv, scaffold


def knn_enrichment(normalised, query_rows, member_mask, universe_size):
    """Enrichment of a set among its own members' nearest neighbours.

    The same quantity `04`'s Q1 computes, written so it can be pointed at any set:
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
    log("  * the RAW functional-group odds ratios from 04's fg_enrichment.csv,")
    log("    with the matched column beside them, never instead of them")
    log("  * the seed spread, as evidence the control is not one lucky pairing")

    (RESULTS / "robustness_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'robustness_report.txt'}")


if __name__ == "__main__":
    main()
