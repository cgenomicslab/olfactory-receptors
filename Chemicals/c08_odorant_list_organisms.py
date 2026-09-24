"""The c07 kingdom comparison for one odorant list, with a table of its molecules.

    python c08_odorant_list_organisms.py              # M2OR
    python c08_odorant_list_organisms.py --panel lg   # Leffingwell / GoodScents

Molecules without a recorded organism are left out.

Outputs in results/: <list>_organism_table.csv, <list>_kingdom_stats.csv,
<list>_organism_report.txt
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import chemspace as cs
from c07_organism_source import (compare_mean_shares, lineage_by_rank,
                                 load_organism_field, split_organisms)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"
CACHE = RESULTS / "taxid_cache.json"

# The odour descriptors for the M2OR ligands, so the table says what each one smells of.
# Only M2OR molecules carry these, so the column is mostly empty for the larger list.
ODOURS = HERE / "processed_data" / "m2or_inchi_odors_smiles.csv"

# The two odorant lists, and how to recognise each in the universe.
PANELS = {
    "m2or": {
        "label": "M2OR",
        "column": "in_m2or",
        "stem": "m2or",
    },
    "lg": {
        "label": "Leffingwell / GoodScents",
        "column": None,          # the union of the two source flags
        "stem": "leffingwell_goodscents",
    },
}

# The groups worth a column: a display name, the NCBI rank to look it up at, and the
# taxon name. Bacteria are taken at superkingdom rather than as the two kingdom-rank
# groups NCBI splits them into (Bacillati and Pseudomonadati), because "bacteria" is the
# distinction that means something here and the split would report the same result twice.
GROUPS = {
    "plants": ("kingdom", "Viridiplantae"),
    "animals": ("kingdom", "Metazoa"),
    "fungi": ("kingdom", "Fungi"),
    "bacteria": ("superkingdom", "Bacteria"),
}

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


def group_taxids(ncbi):
    """
    Map each reported group to the (rank, taxid) that identifies it.

    Resolved by name lookup rather than hard-coded ids, so the script carries no table
    of magic numbers to rot when NCBI renumbers something.

    Returns
    -------
    dict
        (rank, taxid) -> display name
    """
    wanted = {name for _, name in GROUPS.values()}
    found = ncbi.get_name_translator(sorted(wanted))
    missing = [name for name in wanted if name not in found]
    if missing:
        raise SystemExit(f"could not resolve taxon names: {missing}")
    return {(rank, found[name][0]): label
            for label, (rank, name) in GROUPS.items()}


def load_panel():
    """The M2OR molecules, their COCONUT organism field, and their odour descriptors."""
    universe = pd.read_parquet(
        RESULTS / "universe.parquet",
        columns=["inchikey", "smiles", "in_m2or", "in_leffingwell", "in_goodscents",
                 "is_odorant", "in_background", "np_classifier_pathway",
                 "MolWt", "Tb_joback"],
    )
    parents = pd.read_parquet(
        RESULTS / "coconut_parents.parquet", columns=["identifier", "inchikey"]
    )
    names = pd.read_csv(
        COCONUT_CSV, usecols=["identifier", "name"], dtype=str,
        on_bad_lines="skip", low_memory=False,
    )
    # The name is only a label for the table, so the first COCONUT entry's is enough.
    names = parents.merge(names, on="identifier").drop_duplicates("inchikey")

    table = universe.merge(names[["inchikey", "name"]], on="inchikey", how="left")
    table = table.merge(load_organism_field().rename("organisms"),
                        left_on="inchikey", right_index=True, how="left")

    # The odour descriptors are keyed by the InChIKey shipped with M2OR, which is the
    # same key the universe carries, so this is a direct join.
    if ODOURS.exists():
        odours = pd.read_csv(ODOURS)[["InChIKey", "Odors"]]
        odours.columns = ["inchikey", "odours"]
        table = table.merge(odours, on="inchikey", how="left")
    else:
        table["odours"] = pd.NA
    return table


def main():
    from ete3 import NCBITaxa

    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=list(PANELS), default="m2or",
                        help="which odorant list to describe")
    args = parser.parse_args()
    panel_spec = PANELS[args.panel]

    ncbi = NCBITaxa()
    table = load_panel()

    log("=" * 74)
    log(f"{panel_spec['label'].upper()} AND ITS SOURCE ORGANISMS")
    log("=" * 74)

    if panel_spec["column"]:
        in_panel = table[panel_spec["column"]] == 1
    else:
        in_panel = (table.in_leffingwell + table.in_goodscents) > 0
    table["in_panel"] = in_panel.astype(int)
    has_organisms = table.organisms.notna()
    log(f"  molecules in the list              {int(in_panel.sum()):,}")
    log(f"    with an organism recorded        {int((in_panel & has_organisms).sum()):,}")
    log(f"    excluded, no source recorded     {int((in_panel & ~has_organisms).sum()):,}")
    log("  Molecules with no recorded source are dropped, not counted as zero: nobody")
    log("  looked, which is different from having no source.")

    if not CACHE.exists():
        raise SystemExit(f"missing {CACHE}; run c07_organism_source.py first")
    mapping = {k: int(v) for k, v in json.loads(CACHE.read_text()).items()}

    # Everything with an organism, so the panel can be compared against the background.
    annotated = table[has_organisms].reset_index(drop=True)
    per_molecule = [split_organisms(v) for v in annotated.organisms]

    used = sorted({mapping[n] for names in per_molecule for n in names if n in mapping})
    lineages = lineage_by_rank(ncbi, used)
    group_lookup = group_taxids(ncbi)
    log(f"\n  organism names resolved to taxa    {len(used):,}")

    # ------------------------------------------------- per-molecule composition
    rows = []
    for row, names in enumerate(per_molecule):
        counts = Counter()
        resolved = 0
        for name in names:
            taxid = mapping.get(name)
            # Same rule as c07: a name counts only if its lineage could be retrieved.
            if taxid is None or taxid not in lineages:
                continue
            resolved += 1
            ancestors = lineages[taxid]
            for rank, ancestor in ancestors.items():
                label = group_lookup.get((rank, ancestor))
                if label is not None:
                    counts[label] += 1

        record = {"n_organisms": len(names), "n_resolved": resolved}
        for label in GROUPS:
            record[f"share_{label}"] = counts[label] / resolved if resolved else np.nan
        record["dominant_group"] = counts.most_common(1)[0][0] if counts else None
        rows.append(record)

    composition = pd.DataFrame(rows)
    annotated = pd.concat([annotated, composition], axis=1)

    # ------------------------------------------------------------- the table
    panel = annotated[annotated.in_panel == 1].copy()
    panel["top_organisms"] = [
        "; ".join(split_organisms(v)[:3]) for v in panel.organisms
    ]
    columns = (["inchikey", "name", "smiles", "odours", "np_classifier_pathway",
                "MolWt", "Tb_joback", "n_organisms", "n_resolved", "dominant_group"]
               + [f"share_{c}" for c in GROUPS] + ["top_organisms"])
    panel = panel[columns].sort_values("n_organisms", ascending=False)
    table_path = RESULTS / f"{panel_spec['stem']}_organism_table.csv"
    panel.to_csv(table_path, index=False)
    log(f"\n  wrote {table_path}  ({len(panel)} molecules)")

    log("\n  what the panel's molecules mostly come from:")
    for kingdom, count in panel.dominant_group.value_counts().items():
        log(f"    {str(kingdom):26s} {count:4d}  ({100 * count / len(panel):.1f}%)")

    log("\n  the ten molecules with the most recorded sources:")
    for _, row in panel.head(10).iterrows():
        smell = str(row.odours)[:34] if pd.notna(row.odours) else "-"
        log(f"    {str(row['name'])[:26]:26s} {row.n_organisms:5d} sources   "
            f"{str(row.dominant_group):8s}   {smell}")

    # ------------------------------------------- M2OR against the background
    log("\n" + "=" * 74)
    log(f"{panel_spec['label'].upper()} AGAINST THE NATURAL-PRODUCT BACKGROUND, BY KINGDOM")
    log("=" * 74)
    log("  Both sides restricted to molecules with a recorded source organism.")

    is_panel = (annotated.in_panel == 1).values
    is_background = (annotated.in_background == 1).values
    usable = annotated.n_resolved.values > 0
    log(f"    {panel_spec['label']:26s} {int((is_panel & usable).sum()):,}")
    log(f"    natural products  {int((is_background & usable).sum()):,}")

    compared = (is_panel | is_background) & usable
    share_columns = [f"share_{label}" for label in GROUPS]
    shares = sparse.csr_matrix(annotated.loc[compared, share_columns].values)
    result = compare_mean_shares(shares, is_panel[compared])

    kingdom_stats = pd.DataFrame({
        "kingdom": list(GROUPS),
        "mean_share_panel": result.mean_share_group.values,
        "mean_share_background": result.mean_share_rest.values,
        "fold_change": result.share_ratio.values,
        "fold_ci_lo": result.ratio_ci_lo.values,
        "fold_ci_hi": result.ratio_ci_hi.values,
        "n_panel": int((is_panel & usable).sum()),
        "n_background": int((is_background & usable).sum()),
        "p": result.p.values,
    })
    kingdom_stats["q"] = cs.benjamini_hochberg(kingdom_stats["p"].values)
    stats_path = RESULTS / f"{panel_spec['stem']}_kingdom_stats.csv"
    kingdom_stats.to_csv(stats_path, index=False)

    log(f"\n    {'kingdom':26s} {'list':>8s} {'others':>8s} {'fold':>7s} "
        f"{'95% CI':>16s}   {'p':>9s}   q")
    for _, row in kingdom_stats.sort_values("fold_change", ascending=False).iterrows():
        interval = f"[{row.fold_ci_lo:.2f}, {row.fold_ci_hi:.2f}]"
        log(f"    {row.kingdom:26s} {100 * row.mean_share_panel:7.1f}% "
            f"{100 * row.mean_share_background:7.1f}% {row.fold_change:7.2f} "
            f"{interval:>16s}   {row.p:9.1e}   {row.q:.1e}")

    log("\n  Permutation test on the mean share (20,000 label shuffles), so the p-value")
    log("  tests the fold change that is shown; the smallest possible p is 5.0e-05.")
    log("  Interval: percentile bootstrap, both groups resampled. Benjamini-Hochberg")
    log(f"  across the {len(GROUPS)} groups.")

    report_path = RESULTS / f"{panel_spec['stem']}_organism_report.txt"
    report_path.write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {report_path}")


if __name__ == "__main__":
    main()
