"""Step 8 -- sweep every COCONUT column for differences between odorants and the rest.

s01 reads ten of COCONUT's forty-four columns, because those were the ones the analysis
needed. This script goes back to the raw file and tests all of the others, so that
nothing informative is sitting unexamined in a column nobody opened. It is a screen, not
a hypothesis: everything it turns up needs reading with the multiple-testing correction
and the curation caveat in mind.

Three kinds of column, three treatments:

  numeric      Cliff's delta, which asks how often a randomly chosen odorant exceeds a
               randomly chosen natural product. Bounded -1 to +1, unaffected by the very
               skewed distributions here, and not fooled by the outliers that would drag
               a difference in means around.
  yes/no       odds ratio with a confidence interval.
  categorical  odds ratio per category, tested only where a category has enough
               molecules behind it to say anything.

Outputs
-------
results/scan_numeric.csv       every numeric column, ranked by effect size
results/scan_categorical.csv   every category of every categorical column
results/scan_report.txt        everything printed below

CPU only, a few minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import chemspace as cs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"

# COCONUT's own numbers. Some duplicate what RDKit gave us, which is useful as a check;
# the rest -- van der Waals volume, formal charge, the Lipinski counts -- are new.
NUMERIC = [
    "total_atom_count", "heavy_atom_count", "molecular_weight", "alogp",
    "topological_polar_surface_area", "rotatable_bond_count",
    "hydrogen_bond_acceptors", "hydrogen_bond_donors",
    "hydrogen_bond_acceptors_lipinski", "hydrogen_bond_donors_lipinski",
    "lipinski_rule_of_five_violations", "aromatic_rings_count",
    "qed_drug_likeliness", "formal_charge", "fractioncsp3",
    "number_of_minimal_rings", "van_der_walls_volume", "np_likeness",
]

BOOLEAN = [
    "contains_sugar", "contains_ring_sugars", "contains_linear_sugars",
    "np_classifier_is_glycoside",
]

# Free-text or many-valued columns, tested category by category.
CATEGORICAL = [
    "annotation_level", "chemical_super_class", "chemical_class",
    "chemical_sub_class", "direct_parent_classification",
    "np_classifier_superclass", "np_classifier_class", "murcko_framework",
]

# Pipe-separated lists rather than a single value.
MULTI_VALUED = ["collections"]

MIN_MOLECULES = 30

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


def cliffs_delta(odorant_values, background_values, sample=200_000, seed=0):
    """
    How often a random odorant exceeds a random natural product, rescaled to -1..+1.

    Zero means the two are interchangeable; +1 means every odorant is larger. Computed
    on a random subsample of the background, because the exact statistic is quadratic
    and the background has 720,000 members.

    Returns
    -------
    float
    """
    odorant_values = odorant_values[np.isfinite(odorant_values)]
    background_values = background_values[np.isfinite(background_values)]
    if len(odorant_values) < 10 or len(background_values) < 10:
        return np.nan

    rng = np.random.default_rng(seed)
    if len(background_values) > sample:
        background_values = rng.choice(background_values, sample, replace=False)

    # Rank-based form: equivalent to the pairwise definition, but n log n rather than n^2.
    combined = np.concatenate([odorant_values, background_values])
    ranks = stats.rankdata(combined)
    rank_sum = ranks[:len(odorant_values)].sum()
    n_a, n_b = len(odorant_values), len(background_values)
    u_statistic = rank_sum - n_a * (n_a + 1) / 2
    return float(2 * u_statistic / (n_a * n_b) - 1)


def odds_ratio_ci(n_odorant, total_odorant, n_background, total_background):
    """Odds ratio with a Woolf 95% interval, half-count corrected."""
    a = n_odorant + 0.5
    b = n_background + 0.5
    c = total_odorant - n_odorant + 0.5
    d = total_background - n_background + 0.5
    odds_ratio = (a * d) / (b * c)
    log_se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(0.975)
    return (float(odds_ratio),
            float(np.exp(np.log(odds_ratio) - z * log_se)),
            float(np.exp(np.log(odds_ratio) + z * log_se)))


def format_p(p_value):
    """Format a p-value, showing a floor rather than a misleading exact zero."""
    return "<1e-300" if p_value <= 0 else f"{p_value:.1e}"


def load_joined():
    """Join the full COCONUT columns onto the universe, via the cached parents table."""
    universe = pd.read_parquet(
        RESULTS / "universe.parquet", columns=["inchikey", "is_odorant", "in_background"]
    )
    parents = pd.read_parquet(
        RESULTS / "coconut_parents.parquet", columns=["identifier", "inchikey"]
    )
    wanted = ["identifier"] + NUMERIC + BOOLEAN + CATEGORICAL + MULTI_VALUED
    raw = pd.read_csv(COCONUT_CSV, usecols=wanted, on_bad_lines="skip", low_memory=False)

    linked = parents.merge(raw, on="identifier", how="left").drop_duplicates("inchikey")
    return universe.merge(
        linked.drop(columns=["identifier"]), on="inchikey", how="inner"
    )


def scan_numeric(table, is_odorant):
    """Effect size and a rank test for every numeric column."""
    log("\n" + "=" * 74)
    log("NUMERIC COLUMNS  (Cliff's delta: +1 = always higher in odorants, 0 = no difference)")
    log("=" * 74)

    rows = []
    for column in NUMERIC:
        if column not in table:
            continue
        values = pd.to_numeric(table[column], errors="coerce").values.astype(float)
        odorant_values = values[is_odorant]
        background_values = values[~is_odorant]
        if np.isfinite(odorant_values).sum() < 10:
            continue

        delta = cliffs_delta(odorant_values, background_values)
        _, p_value = stats.mannwhitneyu(
            odorant_values[np.isfinite(odorant_values)],
            background_values[np.isfinite(background_values)],
            alternative="two-sided",
        )
        rows.append({
            "column": column,
            "median_odorant": np.nanmedian(odorant_values),
            "median_background": np.nanmedian(background_values),
            "cliffs_delta": delta,
            "p": p_value,
            "n_odorant": int(np.isfinite(odorant_values).sum()),
        })

    numeric = pd.DataFrame(rows)
    numeric["q"] = cs.benjamini_hochberg(numeric["p"].values)
    numeric = numeric.reindex(numeric.cliffs_delta.abs().sort_values(ascending=False).index)
    numeric.to_csv(RESULTS / "scan_numeric.csv", index=False)

    log(f"    {'column':34s} {'odorant':>10s} {'others':>10s} {'delta':>7s}   q")
    for _, row in numeric.iterrows():
        log(f"    {row.column:34s} {row.median_odorant:10.2f} {row.median_background:10.2f} "
            f"{row.cliffs_delta:+7.2f}   {format_p(row.q)}")
    return numeric


def scan_flags(table, is_odorant):
    """Odds ratios for the yes/no columns."""
    log("\n" + "=" * 74)
    log("YES / NO COLUMNS")
    log("=" * 74)

    rows = []
    n_odorant, n_background = int(is_odorant.sum()), int((~is_odorant).sum())
    for column in BOOLEAN:
        if column not in table:
            continue
        flag = table[column].map(
            {True: 1, False: 0, "true": 1, "false": 0, "True": 1, "False": 0}
        ).fillna(0).values.astype(bool)

        with_odorant = int((flag & is_odorant).sum())
        with_background = int((flag & ~is_odorant).sum())
        odds_ratio, low, high = odds_ratio_ci(
            with_odorant, n_odorant, with_background, n_background)
        _, p_value = stats.fisher_exact([
            [with_odorant, n_odorant - with_odorant],
            [with_background, n_background - with_background],
        ])
        rows.append({"column": column,
                     "pct_odorant": 100 * with_odorant / n_odorant,
                     "pct_background": 100 * with_background / n_background,
                     "odds_ratio": odds_ratio, "ci_lo": low, "ci_hi": high,
                     "p": p_value})

    flags = pd.DataFrame(rows)
    for _, row in flags.iterrows():
        log(f"    {row.column:34s} {row.pct_odorant:6.1f}% vs {row.pct_background:6.1f}%"
            f"   OR {row.odds_ratio:6.2f}  [{row.ci_lo:5.2f}, {row.ci_hi:6.2f}]"
            f"   {format_p(row.p)}")
    return flags


def scan_categorical(table, is_odorant):
    """Odds ratio for every category with enough molecules behind it."""
    log("\n" + "=" * 74)
    log(f"CATEGORICAL COLUMNS  (categories with >= {MIN_MOLECULES} molecules)")
    log("=" * 74)

    n_odorant, n_background = int(is_odorant.sum()), int((~is_odorant).sum())
    collected = []

    for column in CATEGORICAL + MULTI_VALUED:
        if column not in table:
            continue
        values = table[column].astype(str)

        if column in MULTI_VALUED:
            # One molecule can belong to several collections.
            membership = {}
            for row, cell in enumerate(values):
                if cell in ("nan", ""):
                    continue
                for item in cell.split("|"):
                    item = item.strip()
                    if item:
                        membership.setdefault(item, np.zeros(len(table), bool))[row] = True
            categories = membership
        else:
            categories = {c: (values == c).values
                          for c in values[values != "nan"].unique()}

        rows = []
        for category, present in categories.items():
            with_odorant = int((present & is_odorant).sum())
            with_background = int((present & ~is_odorant).sum())
            if with_odorant + with_background < MIN_MOLECULES:
                continue
            odds_ratio, low, high = odds_ratio_ci(
                with_odorant, n_odorant, with_background, n_background)
            _, p_value = stats.fisher_exact([
                [with_odorant, n_odorant - with_odorant],
                [with_background, n_background - with_background],
            ])
            rows.append({"column": column, "category": category,
                         "n_odorant": with_odorant, "n_background": with_background,
                         "pct_odorant": 100 * with_odorant / n_odorant,
                         "pct_background": 100 * with_background / n_background,
                         "odds_ratio": odds_ratio, "ci_lo": low, "ci_hi": high,
                         "p": p_value})

        if not rows:
            continue
        found = pd.DataFrame(rows)
        found["q"] = cs.benjamini_hochberg(found["p"].values)
        collected.append(found)

        significant = found[found.q < 0.05]
        log(f"\n  {column}: {len(found)} categories tested, {len(significant)} significant")
        for _, row in significant.sort_values("odds_ratio", ascending=False).head(6).iterrows():
            log(f"    + {str(row.category)[:38]:38s} {row.pct_odorant:6.1f}% vs "
                f"{row.pct_background:6.1f}%   OR {row.odds_ratio:7.2f}   q {format_p(row.q)}")
        for _, row in significant.sort_values("odds_ratio").head(3).iterrows():
            log(f"    - {str(row.category)[:38]:38s} {row.pct_odorant:6.1f}% vs "
                f"{row.pct_background:6.1f}%   OR {row.odds_ratio:7.2f}   q {format_p(row.q)}")

    if collected:
        pd.concat(collected).to_csv(RESULTS / "scan_categorical.csv", index=False)


def main():
    table = load_joined()
    is_odorant = (table.is_odorant == 1).values

    log("=" * 74)
    log("COCONUT DESCRIPTOR SWEEP: odorants against the rest")
    log("=" * 74)
    log(f"  molecules joined to raw COCONUT columns: {len(table):,}")
    log(f"    odorants          {int(is_odorant.sum()):,}")
    log(f"    natural products  {int((~is_odorant).sum()):,}")

    scan_numeric(table, is_odorant)
    scan_flags(table, is_odorant)
    scan_categorical(table, is_odorant)

    log("\n  A screen, not a hypothesis. Everything here is one comparison out of many,")
    log("  and COCONUT records what people looked for as much as what exists.")

    (RESULTS / "scan_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'scan_report.txt'}")


if __name__ == "__main__":
    main()
