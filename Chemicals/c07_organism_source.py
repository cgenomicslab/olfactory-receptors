"""Which organisms odorants are isolated from, against other natural products.

Organism names from COCONUT are resolved against NCBI taxonomy (full name, then genus and
species, then genus). For each molecule, the share of its source organisms in each taxon is
computed, and the mean share is compared between odorants and the rest at every rank:
ratio of means, permutation p-value (20,000 shuffles), bootstrap interval, BH within rank.
A second test uses only molecules with a single recorded organism (Fisher's exact test).

Outputs in results/: organism_rank_enrichment.csv, organism_single_source.csv,
organism_report.txt, taxid_cache.json
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

import chemspace as cs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"
CACHE = RESULTS / "taxid_cache.json"

# The ranks worth testing. Finer than genus is mostly strain designations, and coarser
# than superkingdom is not a distinction.
RANKS = ["superkingdom", "kingdom", "phylum", "class", "order", "family", "genus"]

# A taxon needs this many molecules behind it before it is tested, otherwise the table
# fills up with taxa seen in one compound.
MIN_MOLECULES = 20

# Mammals and birds turn up as "source organisms" because a compound was detected in
# breath, milk or body odour, not because the animal makes it. Flagged here, and left
# out of the figures.
HOST_TAXA = {"Primates", "Artiodactyla", "Mammalia", "Chordata", "Homo sapiens",
             "Carnivora", "Rodentia", "Aves"}

# A ratio of means is unstable when both means are near zero: a taxon accounting for
# 0.05% of odorant sources against 0.0005% of the rest scores 100x and means nothing.
# To be listed, a taxon has to account for at least this share on one side or the other.
MIN_SHARE = 0.005

PERMUTATIONS = 20_000
BOOTSTRAP_RESAMPLES = 2_000

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


# ------------------------------------------------------------- organism names
def clean_organism_name(name):
    """
    Reduce one raw organism string to something NCBI might recognise.

    Strips digits, brackets, quotes and strain codes -- COCONUT's organism field also
    contains occasional gene names and free text, and those break the lookup.
    """
    cleaned = re.sub(r"[^A-Za-z .\-]", " ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_organisms(field):
    """
    Split one organisms cell into its individual cleaned names, each listed once.

    A molecule's sources are a set: the same name twice (which happens once strain codes
    are stripped, or when two COCONUT entries are pooled) is one organism, not two.
    """
    names = []
    for part in re.split(r"[|+]", str(field)):
        cleaned = clean_organism_name(part)
        if len(cleaned) > 2:
            names.append(cleaned)
    return list(dict.fromkeys(names))


def load_organism_field():
    """
    The COCONUT organism field for every molecule in the universe, keyed by InChIKey.

    The universe is keyed by the InChIKey c02 recomputed, so the route back to the raw
    COCONUT columns is through the cached parents table and its identifier. Where several
    COCONUT entries share one InChIKey, their organism fields are pooled.

    Returns
    -------
    Series
        InChIKey -> pipe-separated organism field.
    """
    parents = pd.read_parquet(
        RESULTS / "coconut_parents.parquet", columns=["identifier", "inchikey"]
    )
    organisms = pd.read_csv(
        COCONUT_CSV, usecols=["identifier", "organisms"], dtype=str,
        on_bad_lines="skip", low_memory=False,
    )
    linked = parents.merge(organisms, on="identifier", how="inner")
    linked = linked.dropna(subset=["organisms"])
    return linked.groupby("inchikey")["organisms"].agg("|".join)


def translate_names(ncbi, names, chunk=4000):
    """
    Look up many names at once, in batches.

    ete3 builds a literal SQL statement from the names, so a single awkward string can
    take the whole query down. Batching keeps one bad name from costing more than its
    own batch, and the statement short enough for sqlite.
    """
    resolved = {}
    names = [n for n in names if n]
    for start in range(0, len(names), chunk):
        batch = names[start:start + chunk]
        try:
            resolved.update(ncbi.get_name_translator(batch))
        except Exception:
            # Fall back to one at a time, so only the offending name is lost.
            for name in batch:
                try:
                    resolved.update(ncbi.get_name_translator([name]))
                except Exception:
                    pass
    return resolved


def resolve_to_taxids(ncbi, names):
    """
    Map organism names to NCBI taxon ids, in three passes of decreasing precision.

    Returns
    -------
    dict
        name -> taxid, for the names that resolved.
    """
    names = sorted(set(names))
    log(f"  unique organism names: {len(names):,}")

    exact = translate_names(ncbi, names)
    mapping = {name: ids[0] for name, ids in exact.items()}
    log(f"    exact match              {len(mapping):,}")

    # Second pass: genus + species, dropping any strain designation.
    remaining = [n for n in names if n not in mapping]
    binomial = {n: " ".join(n.split()[:2]) for n in remaining if len(n.split()) >= 2}
    found = translate_names(ncbi, sorted(set(binomial.values())))
    added = 0
    for name, short in binomial.items():
        if short in found:
            mapping[name] = found[short][0]
            added += 1
    log(f"    + genus species          {added:,}")

    # Third pass: genus alone.
    remaining = [n for n in names if n not in mapping]
    genus = {n: n.split()[0] for n in remaining if n.split()}
    found = translate_names(ncbi, sorted(set(genus.values())))
    added = 0
    for name, short in genus.items():
        if short in found:
            mapping[name] = found[short][0]
            added += 1
    log(f"    + genus only             {added:,}")
    log(f"    unresolved               {len(names) - len(mapping):,} "
        f"({100 * (1 - len(mapping) / max(len(names), 1)):.1f}%)")
    return mapping


def load_name_mapping(ncbi, per_molecule):
    """
    Name -> taxid, from the cache where possible.

    The cache only holds names that resolved, so any name not in it -- new, or one that
    failed before -- is looked up again and the cache extended. Failing names cost a
    few seconds per run; a name that has become resolvable is never silently skipped.
    """
    all_names = {n for names in per_molecule for n in names}
    mapping = {}
    if CACHE.exists():
        mapping = {k: int(v) for k, v in json.loads(CACHE.read_text()).items()}
        log(f"  cached name -> taxid map: {len(mapping):,} entries")

    missing = sorted(all_names - set(mapping))
    if missing:
        log(f"  looking up {len(missing):,} names not in the cache")
        mapping.update(resolve_to_taxids(ncbi, missing))
        CACHE.write_text(json.dumps(mapping))
    resolved = len(all_names & set(mapping))
    log(f"  names resolved: {resolved:,} of {len(all_names):,} "
        f"({100 * resolved / len(all_names):.1f}%)")
    return mapping


def lineage_by_rank(ncbi, taxids):
    """
    For each taxon, the ancestor it has at each rank we care about.

    Returns
    -------
    dict
        taxid -> {rank: ancestor_taxid}
    """
    out = {}
    for taxid in taxids:
        try:
            lineage = ncbi.get_lineage(taxid)
            ranks = ncbi.get_rank(lineage)
        except Exception:
            continue
        out[taxid] = {rank: node for node, rank in ranks.items() if rank in RANKS}
    return out


def build_count_matrix(per_molecule, mapping, lineages, rank, n_molecules):
    """
    Count, for every molecule, how many of its source organisms fall in each taxon.

    Returns a sparse matrix rather than a dense one: at genus level there are thousands
    of taxa and 269,000 molecules, and almost every entry is zero.

    Returns
    -------
    counts : scipy sparse matrix, (n_molecules, n_taxa)
    taxa : list of int
        Column order.
    resolved_per_molecule : ndarray
        How many of each molecule's organisms resolved to a taxon with a lineage. This
        is the denominator of every share, at every rank.
    """
    taxon_index = {}
    rows, cols, values = [], [], []
    resolved_per_molecule = np.zeros(n_molecules)

    for row, names in enumerate(per_molecule):
        counts = Counter()
        resolved = 0
        for name in names:
            taxid = mapping.get(name)
            if taxid is None or taxid not in lineages:
                continue
            ancestor = lineages[taxid].get(rank)
            resolved += 1
            if ancestor is not None:
                counts[ancestor] += 1
        resolved_per_molecule[row] = resolved
        for taxon, count in counts.items():
            if taxon not in taxon_index:
                taxon_index[taxon] = len(taxon_index)
            rows.append(row)
            cols.append(taxon_index[taxon])
            values.append(count)

    taxa = [t for t, _ in sorted(taxon_index.items(), key=lambda kv: kv[1])]
    counts = sparse.coo_matrix((values, (rows, cols)),
                               shape=(n_molecules, len(taxa))).tocsr()
    return counts, taxa, resolved_per_molecule


# --------------------------------------------------------------- the statistic
def _group_sums(shares, row_sets):
    """
    Column sums of `shares` over several sets of rows at once.

    Each set of row indices becomes one row of a sparse selection matrix (repeated
    indices add up, which is what a bootstrap resample needs), so a single sparse
    product gives the sum over every set.

    Returns
    -------
    ndarray, (n_sets, n_columns)
    """
    n_sets, set_size = row_sets.shape
    selection = sparse.csr_matrix(
        (np.ones(row_sets.size), row_sets.ravel(),
         np.arange(0, row_sets.size + 1, set_size)),
        shape=(n_sets, shares.shape[0]),
    )
    return np.asarray((selection @ shares).todense())


def compare_mean_shares(shares, in_group, permutations=PERMUTATIONS,
                        resamples=BOOTSTRAP_RESAMPLES, seed=0, batch=500):
    """
    Mean share of each column in one group against the rest, with a permutation test.

    Parameters
    ----------
    shares : scipy sparse matrix, (n_molecules, n_columns)
        For each molecule, the fraction of its source organisms in each taxon.
    in_group : ndarray of bool
        True for the molecules in the group (the odorants).
    permutations : int
        Label shuffles for the p-value.
    resamples : int
        Bootstrap resamples for the interval on the ratio.
    seed : int
    batch : int
        Shuffles or resamples done per sparse product; only affects memory.

    Returns
    -------
    DataFrame
        One row per column: mean_share_group, mean_share_rest, share_ratio,
        ratio_ci_lo, ratio_ci_hi, p.
    """
    rng = np.random.default_rng(seed)
    shares = sparse.csr_matrix(shares, dtype=float)
    group_rows = np.flatnonzero(in_group)
    rest_rows = np.flatnonzero(~in_group)
    n_total, n_group, n_rest = len(in_group), len(group_rows), len(rest_rows)

    column_total = np.asarray(shares.sum(axis=0)).ravel()
    group_sum = np.asarray(shares[group_rows].sum(axis=0)).ravel()
    mean_group = group_sum / n_group
    mean_rest = (column_total - group_sum) / n_rest
    observed = mean_group - mean_rest

    # Permutation: any n_group molecules could have carried the odorant label. The sum
    # over the shuffled group fixes the sum over the rest, so one product per shuffle.
    exceed = np.zeros(len(column_total))
    for start in range(0, permutations, batch):
        size = min(batch, permutations - start)
        shuffled = np.stack([rng.choice(n_total, n_group, replace=False)
                             for _ in range(size)])
        sums = _group_sums(shares, shuffled)
        difference = sums / n_group - (column_total - sums) / n_rest
        # A tolerance, so a shuffle that ties the observed value counts as reaching it.
        exceed += (np.abs(difference) >= np.abs(observed) - 1e-12).sum(axis=0)
    p_value = (exceed + 1) / (permutations + 1)

    # Bootstrap: both groups resampled with replacement, the ratio recomputed each time.
    ratios = []
    for start in range(0, resamples, batch // 5):
        size = min(batch // 5, resamples - start)
        group_draw = group_rows[rng.integers(0, n_group, size=(size, n_group))]
        rest_draw = rest_rows[rng.integers(0, n_rest, size=(size, n_rest))]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append((_group_sums(shares, group_draw) / n_group)
                          / (_group_sums(shares, rest_draw) / n_rest))
    ratios = np.vstack(ratios)
    ratios[~np.isfinite(ratios)] = np.nan
    low, high = np.nanpercentile(ratios, [2.5, 97.5], axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(mean_rest > 0, mean_group / mean_rest, np.nan)
    return pd.DataFrame({
        "mean_share_group": mean_group, "mean_share_rest": mean_rest,
        "share_ratio": ratio, "ratio_ci_lo": low, "ratio_ci_hi": high, "p": p_value,
    })


def compositional_test(count_matrices, is_odorant, ncbi):
    """
    Compare the share of each molecule's sources that fall in a taxon, at every rank.

    This is the test that survives the count confound: a molecule recorded from 500
    plants and one recorded from 1 plant both score 1.0 for Viridiplantae, so a molecule
    with many sources cannot inflate every taxon at once. All ranks are tested in one
    pass, because the molecules are the same at every rank and so are the shuffles.

    Returns
    -------
    DataFrame
        One row per (rank, taxon), with Benjamini-Hochberg q-values within each rank.
    """
    resolved = next(iter(count_matrices.values()))[2]
    usable = resolved > 0

    blocks, labels = [], []
    for rank, (counts, taxa, _) in count_matrices.items():
        counts = counts[usable]
        molecules_with = np.asarray((counts > 0).sum(axis=0)).ravel()
        tested = np.flatnonzero(molecules_with >= MIN_MOLECULES)
        if not len(tested):
            continue
        # Divide each molecule's counts by its number of resolved organisms.
        shares = sparse.diags(1 / resolved[usable]) @ counts[:, tested]
        blocks.append(shares)
        labels += [(rank, taxa[column], int(molecules_with[column])) for column in tested]

    log(f"\n  permutation test on {len(labels):,} taxa over "
        f"{int(usable.sum()):,} molecules ({PERMUTATIONS:,} shuffles) ...")
    result = compare_mean_shares(sparse.hstack(blocks).tocsr(), is_odorant[usable])
    result.insert(0, "rank", [rank for rank, _, _ in labels])
    result.insert(1, "taxid", [taxid for _, taxid, _ in labels])
    result["n_molecules"] = [n for _, _, n in labels]
    # The two group sizes the shares are averaged over, the same for every taxon.
    result["n_odorant_molecules"] = int(is_odorant[usable].sum())
    result["n_background_molecules"] = int((~is_odorant[usable]).sum())
    result = result.rename(columns={"mean_share_group": "mean_share_odorant",
                                    "mean_share_rest": "mean_share_background"})

    result["q"] = np.nan
    for rank in RANKS:
        at_rank = result["rank"] == rank
        if at_rank.any():
            result.loc[at_rank, "q"] = cs.benjamini_hochberg(result.loc[at_rank, "p"].values)

    names = ncbi.get_taxid_translator(list(result.taxid))
    result["taxon"] = [names.get(t, str(t)) for t in result.taxid]
    return result


def single_source_test(counts, taxa, resolved, is_odorant, rank, ncbi):
    """
    The same question restricted to molecules recorded from exactly one organism.

    A much smaller set, but the count confound cannot arise: each molecule contributes
    to exactly one taxon, so the comparison is a clean one between two compositions.
    Fisher's exact test, odds ratio and 95% interval from the same table.
    """
    single = resolved == 1
    odorant_mask = is_odorant[single]
    n_odorant = int(odorant_mask.sum())
    n_background = int((~odorant_mask).sum())
    if n_odorant < MIN_MOLECULES:
        return pd.DataFrame(), n_odorant, n_background

    present_matrix = counts[single] > 0
    rows = []
    for column, taxon in enumerate(taxa):
        present = np.asarray(present_matrix[:, column].todense()).ravel()
        with_odorant = int((present & odorant_mask).sum())
        with_background = int((present & ~odorant_mask).sum())
        if with_odorant + with_background < MIN_MOLECULES:
            continue

        _, p_value = stats.fisher_exact([
            [with_odorant, n_odorant - with_odorant],
            [with_background, n_background - with_background],
        ])
        odds_ratio, low, high = cs.odds_ratio_ci(with_odorant, n_odorant,
                                                 with_background, n_background)
        rows.append({
            "rank": rank, "taxid": taxon,
            "pct_odorant": 100 * with_odorant / max(n_odorant, 1),
            "pct_background": 100 * with_background / max(n_background, 1),
            "odds_ratio": odds_ratio, "ci_lo": low, "ci_hi": high,
            "n_odorant": with_odorant, "n_background": with_background, "p": p_value,
        })

    if not rows:
        return pd.DataFrame(), n_odorant, n_background

    table = pd.DataFrame(rows)
    table["q"] = cs.benjamini_hochberg(table["p"].values)
    names = ncbi.get_taxid_translator(list(table.taxid))
    table["taxon"] = [names.get(t, str(t)) for t in table.taxid]
    return table, n_odorant, n_background


# --------------------------------------------------------------------- report
def report_rank(rank, shares, singles, n_single_odorant, n_single_background):
    """Print the substantial taxa at one rank, and the single-source control."""
    log("\n" + "=" * 74)
    log(f"{rank.upper()}")
    log("=" * 74)

    at_rank = shares[shares["rank"] == rank]
    if not at_rank.empty:
        significant = at_rank[at_rank.q < 0.05].dropna(subset=["share_ratio"])
        substantial = significant[
            significant[["mean_share_odorant", "mean_share_background"]].max(axis=1)
            >= MIN_SHARE
        ]
        log(f"  share of a molecule's sources in each taxon "
            f"({len(at_rank)} taxa tested, {len(significant)} significant, "
            f"{len(substantial)} accounting for >= {100 * MIN_SHARE:.1f}% of sources)")
        log(f"    {'taxon':30s} {'odorants':>9s} {'others':>9s} {'ratio':>7s} "
            f"{'95% CI':>15s}   q")
        ordered = substantial.sort_values("share_ratio", ascending=False)
        # Head and tail overlap when a rank has few taxa, so show everything then.
        shown = (list(ordered.iterrows()) if len(ordered) <= 12
                 else list(ordered.head(8).iterrows()) + list(ordered.tail(4).iterrows()))
        for _, row in shown:
            note = "  <- host, not maker?" if row.taxon in HOST_TAXA else ""
            interval = f"[{row.ratio_ci_lo:.2f}, {row.ratio_ci_hi:.2f}]"
            log(f"    {row.taxon[:30]:30s} {100 * row.mean_share_odorant:8.1f}% "
                f"{100 * row.mean_share_background:8.1f}% {row.share_ratio:7.2f} "
                f"{interval:>15s}   {row.q:.1e}{note}")

    if not singles.empty:
        log(f"\n  control: molecules with exactly one recorded organism "
            f"({n_single_odorant:,} odorants vs {n_single_background:,} others)")
        ordered = singles[singles.q < 0.05].sort_values("odds_ratio", ascending=False)
        enriched = ordered[ordered.odds_ratio > 1].head(5)
        depleted = ordered[ordered.odds_ratio < 1].tail(3).iloc[::-1]
        for sign, subset in [("+", enriched), ("-", depleted)]:
            for _, row in subset.iterrows():
                note = "  <- host, not maker?" if row.taxon in HOST_TAXA else ""
                log(f"    {sign} {row.taxon[:30]:30s} {row.pct_odorant:6.1f}% vs "
                    f"{row.pct_background:6.1f}%   OR {row.odds_ratio:7.2f}"
                    f"   q {row.q:.1e}{note}")


def main():
    from ete3 import NCBITaxa

    ncbi = NCBITaxa()
    RESULTS.mkdir(exist_ok=True)

    log("=" * 74)
    log("ORGANISM SOURCE: where odorants come from, against the rest of COCONUT")
    log("=" * 74)

    universe = pd.read_parquet(RESULTS / "universe.parquet",
                               columns=["inchikey", "is_odorant"])
    table = universe.merge(load_organism_field().rename("organisms"),
                           left_on="inchikey", right_index=True, how="inner")
    is_odorant = (table.is_odorant == 1).values
    log(f"  molecules with an organism recorded: {len(table):,}")
    log(f"    odorants          {int(is_odorant.sum()):,}")
    log(f"    natural products  {int((~is_odorant).sum()):,}")

    log("\nresolving organism names against NCBI taxonomy")
    per_molecule = [split_organisms(v) for v in table.organisms]
    log(f"  organism mentions: {sum(len(n) for n in per_molecule):,}")
    mapping = load_name_mapping(ncbi, per_molecule)

    used_taxids = sorted({mapping[n] for names in per_molecule for n in names
                          if n in mapping})
    log(f"\n  distinct taxa referenced: {len(used_taxids):,}")
    lineages = lineage_by_rank(ncbi, used_taxids)
    log(f"  lineages retrieved      : {len(lineages):,}")

    log("\n  organisms recorded per molecule (the reason for the method below):")
    counts_per_molecule = np.array([len(n) for n in per_molecule])
    for label, mask in [("odorants", is_odorant), ("natural products", ~is_odorant)]:
        values = counts_per_molecule[mask]
        log(f"    {label:18s} median {np.median(values):5.0f}   mean {values.mean():6.1f}")
    log("    Odorants are recorded from far more organisms, so a presence/absence test")
    log("    would call every taxon enriched. Shares per molecule are used instead.")

    count_matrices = {rank: build_count_matrix(per_molecule, mapping, lineages, rank,
                                               len(table))
                      for rank in RANKS}
    resolved = count_matrices[RANKS[0]][2]
    log(f"\n  molecules with at least one resolved organism: {int((resolved > 0).sum()):,} "
        f"({int((is_odorant & (resolved > 0)).sum()):,} odorants)")

    shares = compositional_test(count_matrices, is_odorant, ncbi)
    shares.to_csv(RESULTS / "organism_rank_enrichment.csv", index=False)

    single_source = []
    for rank in RANKS:
        counts, taxa, resolved = count_matrices[rank]
        singles, n_single_odorant, n_single_background = single_source_test(
            counts, taxa, resolved, is_odorant, rank, ncbi)
        if not singles.empty:
            single_source.append(singles)
        report_rank(rank, shares, singles, n_single_odorant, n_single_background)

    if single_source:
        pd.concat(single_source).to_csv(
            RESULTS / "organism_single_source.csv", index=False)

    log(f"\n  p-values: permutation test, {PERMUTATIONS:,} label shuffles; the smallest")
    log(f"  possible value is {1 / (PERMUTATIONS + 1):.1e}. Intervals: percentile "
        f"bootstrap, {BOOTSTRAP_RESAMPLES:,} resamples.")
    log("  Benjamini-Hochberg within each rank.")
    log("\n  Reminder: COCONUT records where people looked. Plants are over-studied for")
    log("  volatiles and Streptomyces for antibiotics, so this compares two literatures")
    log("  as well as two chemistries.")

    (RESULTS / "organism_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'organism_report.txt'}")


if __name__ == "__main__":
    main()
