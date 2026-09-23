"""Step 7 -- which organisms make odorants, and how do they differ from the rest?

COCONUT records, for most of its molecules, the organisms the compound was isolated
from. That is a different kind of information from everything else in this folder: not
what an odorant is made of, but what makes it. If odour chemistry is drawn from a
particular part of the tree of life, that is worth knowing, and it is the sort of claim
an ecology and evolution audience will care about more than a functional-group count.

The organism strings are free text -- "Streptomyces sp. BCC 21795", pipe-separated,
sometimes with co-cultures joined by "+". They are resolved against NCBI taxonomy in
three passes: the full cleaned name, then the first two words (genus species), then the
genus alone. About 94% resolve. Names that survive none of those are dropped and counted.

The obvious test -- what fraction of odorant molecules are recorded in taxon X -- does
not work here, and it is worth saying why. Odorants are recorded from 11.7x as many
organisms as the average natural product (mean 50.4 against 4.3): they are common,
well-studied compounds that turn up everywhere. Under a presence/absence test that alone
makes every taxon look enriched, including mutually exclusive ones.

So the primary test is compositional. For each molecule we ask what FRACTION of its
source organisms fall in a given taxon, which does not care how many sources it has, and
compare those fractions between odorants and the rest. Alongside it is a stricter
control: the molecules recorded from exactly one organism, where the source is
unambiguous and the count confound cannot arise at all.

Caveat that belongs with every number here: this is a record of what people have looked
for and where. Plants are over-studied for volatiles and Streptomyces for antibiotics,
so the comparison is between two literatures as much as between two chemistries. It is
still informative, because both sides are drawn from the same database and the same
curation, but it is not a survey of nature.

Outputs
-------
results/organism_rank_enrichment.csv   every taxon tested, at every rank
results/organism_kingdom_summary.csv   the top-level composition
results/organism_report.txt            everything printed below
results/taxid_cache.json               resolved names, so re-runs are fast

CPU only, ~20 min on the first run and a couple of minutes afterwards.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
# breath, milk or body odour, not because the animal makes it. Flagged rather than
# dropped, since the same is arguably true of anything an animal eats.
HOST_TAXA = {"Primates", "Artiodactyla", "Mammalia", "Chordata", "Homo sapiens",
             "Carnivora", "Rodentia", "Aves"}

# A ratio of means is unstable when both means are near zero: a taxon accounting for
# 0.05% of odorant sources against 0.0005% of the rest scores 100x and means nothing.
# To be listed, a taxon has to account for at least this share on one side or the other.
MIN_SHARE = 0.005

_report_lines = []


def log(message=""):
    """Print a line and keep it, so the whole run can be saved to a report file."""
    print(message, flush=True)
    _report_lines.append(str(message))


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
    """Split one organisms cell into its individual cleaned names."""
    names = []
    for part in re.split(r"[|+]", str(field)):
        cleaned = clean_organism_name(part)
        if len(cleaned) > 2:
            names.append(cleaned)
    return names


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


def load_molecules_with_organisms():
    """
    Join COCONUT's organism field onto the universe.

    The universe is keyed by an InChIKey that RDKit recomputed, so the route back to the
    raw COCONUT columns is through the cached parents table and its identifier.
    """
    universe = pd.read_parquet(
        RESULTS / "universe.parquet",
        columns=["inchikey", "is_odorant", "in_background"],
    )
    parents = pd.read_parquet(
        RESULTS / "coconut_parents.parquet", columns=["identifier", "inchikey"]
    )
    organisms = pd.read_csv(
        COCONUT_CSV, usecols=["identifier", "organisms"], dtype=str,
        on_bad_lines="skip", low_memory=False,
    )

    linked = parents.merge(organisms, on="identifier", how="left")
    linked = linked.dropna(subset=["organisms"]).drop_duplicates("inchikey")

    table = universe.merge(linked[["inchikey", "organisms"]], on="inchikey", how="inner")
    return table


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
        How many of each molecule's organisms resolved at all.
    """
    from scipy.sparse import coo_matrix

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
    counts = coo_matrix((values, (rows, cols)),
                        shape=(n_molecules, len(taxa))).tocsc()
    return counts, taxa, resolved_per_molecule


def compositional_test(counts, taxa, resolved, is_odorant, rank, ncbi):
    """
    Compare the share of each molecule's sources that fall in a taxon.

    This is the test that survives the count confound: a molecule recorded from 500
    plants and one recorded from 1 plant both score 1.0 for Viridiplantae, so a molecule
    with many sources cannot inflate every taxon at once.

    Cliff's delta is the effect size -- the chance that a random odorant has a higher
    share than a random natural product, rescaled to -1..+1.
    """
    usable = resolved > 0
    n_usable = int(usable.sum())
    odorant_mask = is_odorant[usable]

    rows = []
    for column, taxon in enumerate(taxa):
        column_counts = np.asarray(counts[:, column].todense()).ravel()[usable]
        n_with = int((column_counts > 0).sum())
        if n_with < MIN_MOLECULES:
            continue

        share = column_counts / resolved[usable]
        odorant_share = share[odorant_mask]
        background_share = share[~odorant_mask]

        try:
            _, p_value = stats.mannwhitneyu(odorant_share, background_share,
                                            alternative="two-sided")
        except ValueError:
            continue

        # Deliberately not Cliff's delta. These shares are mostly zero -- a molecule has
        # no sources at all in most taxa -- and on zero-inflated data the rank statistic
        # can come out positive while the mean share is lower, which reads as a flat
        # contradiction. The ratio of mean shares says what we actually mean: what
        # multiple of an odorant's sources fall in this taxon, against the rest.
        mean_odorant = float(odorant_share.mean())
        mean_background = float(background_share.mean())
        ratio = mean_odorant / mean_background if mean_background > 0 else np.nan

        rows.append({
            "rank": rank, "taxid": taxon,
            "mean_share_odorant": mean_odorant,
            "mean_share_background": mean_background,
            "share_ratio": ratio,
            "n_molecules": n_with, "p": p_value,
        })

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    table["q"] = cs.benjamini_hochberg(table["p"].values)
    names = ncbi.get_taxid_translator(list(table.taxid))
    table["taxon"] = [names.get(t, str(t)) for t in table.taxid]
    return table


def single_source_test(counts, taxa, resolved, is_odorant, rank, ncbi):
    """
    The same question restricted to molecules recorded from exactly one organism.

    A much smaller set, but the count confound cannot arise: each molecule contributes
    to exactly one taxon, so the comparison is a clean one between two compositions.
    """
    single = resolved == 1
    odorant_mask = is_odorant[single]
    n_odorant = int(odorant_mask.sum())
    n_background = int((~odorant_mask).sum())
    if n_odorant < MIN_MOLECULES:
        return pd.DataFrame(), n_odorant, n_background

    rows = []
    for column, taxon in enumerate(taxa):
        present = np.asarray(counts[:, column].todense()).ravel()[single] > 0
        with_odorant = int((present & odorant_mask).sum())
        with_background = int((present & ~odorant_mask).sum())
        if with_odorant + with_background < MIN_MOLECULES:
            continue

        odds_ratio, p_value = stats.fisher_exact([
            [with_odorant, n_odorant - with_odorant],
            [with_background, n_background - with_background],
        ])
        rows.append({
            "rank": rank, "taxid": taxon,
            "pct_odorant": 100 * with_odorant / max(n_odorant, 1),
            "pct_background": 100 * with_background / max(n_background, 1),
            "odds_ratio": odds_ratio, "n_odorant": with_odorant,
            "n_background": with_background, "p": p_value,
        })

    if not rows:
        return pd.DataFrame(), n_odorant, n_background

    table = pd.DataFrame(rows)
    table["q"] = cs.benjamini_hochberg(table["p"].values)
    names = ncbi.get_taxid_translator(list(table.taxid))
    table["taxon"] = [names.get(t, str(t)) for t in table.taxid]
    return table, n_odorant, n_background


def main():
    from ete3 import NCBITaxa

    ncbi = NCBITaxa()
    RESULTS.mkdir(exist_ok=True)

    log("=" * 74)
    log("ORGANISM SOURCE: where odorants come from, against the rest of COCONUT")
    log("=" * 74)

    table = load_molecules_with_organisms()
    is_odorant = (table.is_odorant == 1).values
    log(f"  molecules with an organism recorded: {len(table):,}")
    log(f"    odorants          {int(is_odorant.sum()):,}")
    log(f"    natural products  {int((~is_odorant).sum()):,}")

    # ------------------------------------------------------------ name resolution
    log("\nresolving organism names against NCBI taxonomy")
    per_molecule = [split_organisms(v) for v in table.organisms]
    all_names = [n for names in per_molecule for n in names]
    log(f"  organism mentions: {len(all_names):,}")

    if CACHE.exists():
        mapping = {k: int(v) for k, v in json.loads(CACHE.read_text()).items()}
        log(f"  using cached name -> taxid map ({len(mapping):,} entries)")
    else:
        mapping = resolve_to_taxids(ncbi, all_names)
        CACHE.write_text(json.dumps(mapping))
        log(f"  cached to {CACHE.name}")

    # ---------------------------------------------------------------- lineages
    used_taxids = sorted({mapping[n] for names in per_molecule for n in names
                          if n in mapping})
    log(f"\n  distinct taxa referenced: {len(used_taxids):,}")
    lineages = lineage_by_rank(ncbi, used_taxids)
    log(f"  lineages retrieved      : {len(lineages):,}")

    # ------------------------------------------------------------ the comparison
    log("\n  organisms recorded per molecule (the reason for the method below):")
    counts_per_molecule = np.array([len(n) for n in per_molecule])
    for label, mask in [("odorants", is_odorant), ("natural products", ~is_odorant)]:
        values = counts_per_molecule[mask]
        log(f"    {label:18s} median {np.median(values):5.0f}   mean {values.mean():6.1f}")
    log("    Odorants are recorded from far more organisms, so a presence/absence test")
    log("    would call every taxon enriched. Shares per molecule are used instead.")

    compositional, single_source = [], []
    for rank in RANKS:
        counts, taxa, resolved = build_count_matrix(
            per_molecule, mapping, lineages, rank, len(table))
        if not taxa:
            continue

        shares = compositional_test(counts, taxa, resolved, is_odorant, rank, ncbi)
        if not shares.empty:
            compositional.append(shares)

        singles, n_single_odorant, n_single_background = single_source_test(
            counts, taxa, resolved, is_odorant, rank, ncbi)
        if not singles.empty:
            single_source.append(singles)

        log("\n" + "=" * 74)
        log(f"{rank.upper()}")
        log("=" * 74)

        if not shares.empty:
            significant = shares[shares.q < 0.05].dropna(subset=["share_ratio"])
            substantial = significant[
                significant[["mean_share_odorant", "mean_share_background"]].max(axis=1)
                >= MIN_SHARE
            ]
            log(f"  share of a molecule's sources in each taxon "
                f"({len(shares)} taxa tested, {len(significant)} significant, "
                f"{len(substantial)} accounting for >= {100 * MIN_SHARE:.1f}% of sources)")
            log(f"    {'taxon':30s} {'odorants':>9s} {'others':>9s} {'ratio':>7s}   q")
            ordered = substantial.sort_values("share_ratio", ascending=False)
            # Head and tail overlap when a rank has few taxa, so show everything then.
            shown = (list(ordered.iterrows()) if len(ordered) <= 12
                     else list(ordered.head(8).iterrows())
                     + list(ordered.tail(4).iterrows()))
            for _, row in shown:
                note = "  <- host, not maker?" if row.taxon in HOST_TAXA else ""
                log(f"    {row.taxon[:30]:30s} {100 * row.mean_share_odorant:8.1f}% "
                    f"{100 * row.mean_share_background:8.1f}% {row.share_ratio:7.2f}"
                    f"   {row.q:.1e}{note}")

        if not singles.empty:
            log(f"\n  control: molecules with exactly one recorded organism "
                f"({n_single_odorant:,} odorants vs {n_single_background:,} others)")
            ordered = singles[singles.q < 0.05].sort_values(
                "odds_ratio", ascending=False)
            enriched = ordered[ordered.odds_ratio > 1].head(5)
            depleted = ordered[ordered.odds_ratio < 1].tail(3).iloc[::-1]
            for _, row in enriched.iterrows():
                note = "  <- host, not maker?" if row.taxon in HOST_TAXA else ""
                log(f"    + {row.taxon[:30]:30s} {row.pct_odorant:6.1f}% vs "
                    f"{row.pct_background:6.1f}%   OR {row.odds_ratio:7.2f}"
                    f"   q {row.q:.1e}{note}")
            for _, row in depleted.iterrows():
                log(f"    - {row.taxon[:30]:30s} {row.pct_odorant:6.1f}% vs "
                    f"{row.pct_background:6.1f}%   OR {row.odds_ratio:7.2f}"
                    f"   q {row.q:.1e}")

    if compositional:
        pd.concat(compositional).to_csv(
            RESULTS / "organism_rank_enrichment.csv", index=False)
    if single_source:
        pd.concat(single_source).to_csv(
            RESULTS / "organism_single_source.csv", index=False)

    log("\n  Reminder: COCONUT records where people looked. Plants are over-studied for")
    log("  volatiles and Streptomyces for antibiotics, so this compares two literatures")
    log("  as well as two chemistries.")

    (RESULTS / "organism_report.txt").write_text("\n".join(_report_lines) + "\n")
    log(f"\nwrote {RESULTS / 'organism_report.txt'}")


if __name__ == "__main__":
    main()
