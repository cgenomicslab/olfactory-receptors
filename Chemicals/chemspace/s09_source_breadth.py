"""Step 9 -- are odorants really found in more organisms, or does it only look that way?

s07 turned up a pattern before it turned up any taxon: odorants are recorded from far
more source organisms than other natural products (mean 50.4 against 4.3). That is
either a real ecological fact -- volatile compounds like limonene and linalool genuinely
occur across enormous numbers of species -- or an artefact of famous compounds being
looked for everywhere. The two have completely different implications, so this script
tries to separate them.

Four checks, each removing one alternative explanation:

  1  Is it a few promiscuous outliers?  Compare the whole distribution, not the mean.
  2  Is it study effort?  COCONUT records the papers behind each entry, so the number of
     DOIs is a direct measure of how hard people have looked. If the pattern is only
     effort, it should vanish once compared at equal numbers of papers.
  3  Is it molecular size?  Small simple molecules plausibly occur in more species for
     reasons that have nothing to do with smell. Compared within weight bands.
  4  Is it just database membership?  Molecules curated into many collections are the
     well-known ones. Compared within equal collection counts.

Only molecules that carry an organism annotation are used anywhere here, so the
comparison is always like with like.

Outputs
-------
results/source_breadth.csv     the per-band numbers behind every check
results/source_breadth.txt     everything printed below

CPU only, a couple of minutes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


def cliffs_delta(a, b, sample=150_000, seed=0):
    """
    How often a random member of `a` exceeds a random member of `b`, rescaled to -1..+1.

    Zero means the two are interchangeable. Robust to the very heavy tails here, where a
    handful of molecules are recorded from over a thousand organisms.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan

    rng = np.random.default_rng(seed)
    if len(b) > sample:
        b = rng.choice(b, sample, replace=False)

    ranks = stats.rankdata(np.concatenate([a, b]))
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


def load():
    """Molecules with an organism annotation, plus the columns the checks need."""
    universe = pd.read_parquet(
        RESULTS / "universe.parquet",
        columns=["inchikey", "is_odorant", "MolWt"],
    )
    parents = pd.read_parquet(
        RESULTS / "coconut_parents.parquet", columns=["identifier", "inchikey"]
    )
    raw = pd.read_csv(
        COCONUT_CSV,
        usecols=["identifier", "organisms", "dois", "collections", "annotation_level"],
        dtype=str, on_bad_lines="skip", low_memory=False,
    )

    linked = parents.merge(raw, on="identifier", how="left")
    linked = linked.dropna(subset=["organisms"]).drop_duplicates("inchikey")
    table = universe.merge(
        linked[["inchikey", "organisms", "dois", "collections", "annotation_level"]],
        on="inchikey", how="inner",
    )

    def count_items(cell):
        if not isinstance(cell, str) or not cell.strip():
            return 0
        return len([p for p in cell.replace("+", "|").split("|") if p.strip()])

    table["n_organisms"] = table.organisms.map(count_items)
    table["n_papers"] = table.dois.map(count_items)
    table["n_collections"] = table.collections.map(count_items)
    return table


def compare_within_bands(table, band_column, bands, label):
    """
    Compare organism counts between odorants and the rest, inside each band.

    If the whole pattern is explained by `band_column`, the difference should collapse
    once the comparison happens at equal values of it.
    """
    log(f"\n  within bands of {label}:")
    log(f"    {'band':>18s} {'n odor':>7s} {'n other':>9s} "
        f"{'median odor':>12s} {'median other':>13s} {'delta':>7s}")

    rows = []
    is_odorant = table.is_odorant == 1
    for low, high in bands:
        in_band = (table[band_column] >= low) & (table[band_column] < high)
        odorant_counts = table.n_organisms[in_band & is_odorant].values
        other_counts = table.n_organisms[in_band & ~is_odorant].values
        if len(odorant_counts) < 20 or len(other_counts) < 20:
            continue

        delta = cliffs_delta(odorant_counts, other_counts)
        name = f"{low:g}-{high:g}" if np.isfinite(high) else f">={low:g}"
        log(f"    {name:>18s} {len(odorant_counts):7,d} {len(other_counts):9,d} "
            f"{np.median(odorant_counts):12.0f} {np.median(other_counts):13.0f} "
            f"{delta:+7.2f}")
        rows.append({"check": label, "band": name,
                     "n_odorant": len(odorant_counts), "n_other": len(other_counts),
                     "median_odorant": float(np.median(odorant_counts)),
                     "median_other": float(np.median(other_counts)),
                     "cliffs_delta": delta})
    return rows


def main():
    table = load()
    is_odorant = table.is_odorant == 1

    log("=" * 74)
    log("ARE ODORANTS REALLY FOUND IN MORE ORGANISMS?")
    log("=" * 74)
    log(f"  molecules with an organism annotation: {len(table):,}")
    log(f"    odorants          {int(is_odorant.sum()):,}")
    log(f"    natural products  {int((~is_odorant).sum()):,}")

    # ---------------------------------------------- 1  the whole distribution
    log("\n" + "-" * 74)
    log("CHECK 1  the whole distribution, not the mean")
    log("-" * 74)
    log(f"    {'percentile':>12s} {'odorants':>10s} {'others':>10s}")
    for percentile in [10, 25, 50, 75, 90, 99]:
        a = np.percentile(table.n_organisms[is_odorant], percentile)
        b = np.percentile(table.n_organisms[~is_odorant], percentile)
        log(f"    {percentile:11d}% {a:10.0f} {b:10.0f}")

    overall = cliffs_delta(table.n_organisms[is_odorant],
                           table.n_organisms[~is_odorant])
    log(f"\n    Cliff's delta overall: {overall:+.2f}")
    log(f"    odorants recorded from a single organism : "
        f"{100 * (table.n_organisms[is_odorant] == 1).mean():.1f}%")
    log(f"    others  recorded from a single organism  : "
        f"{100 * (table.n_organisms[~is_odorant] == 1).mean():.1f}%")

    rows = []

    # ---------------------------------------------------- 2  study effort
    log("\n" + "-" * 74)
    log("CHECK 2  is it just that odorants are better studied?")
    log("-" * 74)
    log(f"    papers per molecule, median: odorants "
        f"{table.n_papers[is_odorant].median():.0f}, "
        f"others {table.n_papers[~is_odorant].median():.0f}")
    rows += compare_within_bands(
        table, "n_papers", [(1, 2), (2, 3), (3, 6), (6, 11), (11, np.inf)],
        "papers behind the entry")

    # ---------------------------------------------------------- 3  size
    log("\n" + "-" * 74)
    log("CHECK 3  is it just that odorants are small?")
    log("-" * 74)
    rows += compare_within_bands(
        table, "MolWt", [(0, 150), (150, 250), (250, 350), (350, 500), (500, np.inf)],
        "molecular weight")

    # ------------------------------------------------- 4  database membership
    log("\n" + "-" * 74)
    log("CHECK 4  is it just that odorants are in more databases?")
    log("-" * 74)
    rows += compare_within_bands(
        table, "n_collections", [(1, 2), (2, 3), (3, 5), (5, np.inf)],
        "collections listing it")

    pd.DataFrame(rows).to_csv(RESULTS / "source_breadth.csv", index=False)

    log("\n" + "=" * 74)
    log("READING THIS")
    log("=" * 74)
    log("  If the difference survives inside every band of papers, weight and database")
    log("  membership, then breadth of occurrence is a property of odour chemistry and")
    log("  not of how hard anyone looked. If it collapses in any one of them, that")
    log("  variable was the explanation.")

    (RESULTS / "source_breadth.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'source_breadth.txt'}")


if __name__ == "__main__":
    main()
