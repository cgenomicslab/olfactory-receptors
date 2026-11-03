"""Share of each molecule's source organisms that live in water.

Each organism is labelled aquatic, terrestrial, host (mammals and birds, not counted) or
unassigned, by walking up its NCBI lineage until it reaches a listed clade.
share_aquatic = aquatic / (aquatic + terrestrial). Used by Supplementary_Material/aquatic_origin.

Needs data/taxa.sqlite (c00) and results/panel_universe_join.csv (c09).
Outputs in results/: aquatic_sources.csv, aquatic_sources_panel.csv, aquatic_taxa_seen.csv,
aquatic_sources_report.txt
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import chemspace as cs
from c07_organism_source import (load_organism_field, read_name_cache, resolve_to_taxids,
                                 split_organisms)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"
PANEL_JOIN = RESULTS / "panel_universe_join.csv"

# COCONUT collections that are marine by construction. Used only as an independent check
# on the taxonomy: it is a property of the molecule's literature record, not of the
# occurrence, so it never feeds share_aquatic.
MARINE_COLLECTIONS = {"CMNPD", "Marine Natural Products",
                      "Seaweed Metabolite Database (SWMD)"}

# Clades that live in water, resolved to taxids by name at run time.
AQUATIC = {
    # --- algae and the aquatic protist lineages ---------------------------------
    "Rhodophyta", "Ochrophyta", "Phaeophyceae", "Bacillariophyta", "Chrysophyceae",
    "Chlorophyta", "Dinophyceae", "Haptophyta", "Cryptophyta", "Glaucocystophyceae",
    "Charophyceae", "Zygnematophyceae", "Klebsormidiophyceae", "Coleochaetophyceae",
    "Euglenida", "Labyrinthulomycetes",
    # --- animal phyla that are entirely aquatic ---------------------------------
    "Porifera", "Cnidaria", "Ctenophora", "Echinodermata", "Bryozoa", "Brachiopoda",
    "Chaetognatha", "Hemichordata", "Placozoa", "Xenacoelomorpha", "Entoprocta",
    "Phoronida", "Tunicata", "Cephalochordata",
    # --- fish, crustaceans, marine mammals and reptiles -------------------------
    "Chondrichthyes", "Actinopterygii", "Cladistia", "Coelacanthimorpha", "Dipnoi",
    "Hyperoartia", "Myxini", "Petromyzontiformes",
    # NCBI has no taxon named Crustacea. Pancrustacea stands for it: the hexapods inside
    # it are labelled terrestrial lower down the lineage, so they never reach this rule.
    "Pancrustacea", "Cetacea", "Sirenia", "Pinnipedia", "Cheloniidae", "Dermochelyidae",
    # --- land plants that grow in water -----------------------------------------
    # Seagrasses first, then the freshwater macrophytes and the aquatic ferns. This is
    # the block that stops COCONUT's plant-heavy annotation from making every plant
    # molecule terrestrial by default.
    "Zosteraceae", "Posidoniaceae", "Cymodoceaceae", "Ruppiaceae", "Hydrocharitaceae",
    "Nymphaeaceae", "Nelumbonaceae", "Cabombaceae", "Ceratophyllaceae",
    "Potamogetonaceae", "Lemnoideae", "Pontederiaceae", "Alismataceae", "Butomaceae",
    "Aponogetonaceae", "Juncaginaceae", "Menyanthaceae", "Podostemaceae",
    "Salviniaceae", "Marsileaceae", "Isoetaceae", "Ricciaceae",
}

# Land trees rooted in the intertidal. Counted apart from both buckets so the decision is
# visible in the output rather than buried in a set literal.
MANGROVE = {"Rhizophoraceae", "Avicennia", "Sonneratia", "Laguncularia", "Conocarpus"}

# Everything on land. Checked only after the aquatic sets, so the aquatic members of these
# clades -- seagrasses inside Embryophyta, whales inside Artiodactyla -- are already gone.
TERRESTRIAL = {"Embryophyta", "Insecta", "Arachnida", "Myriapoda", "Onychophora",
               "Collembola", "Diplura", "Protura", "Amphibia", "Testudines", "Squamata",
               "Crocodylia"}

# Detected in, not made by. c07's convention.
HOST = {"Homo sapiens", "Primates", "Artiodactyla", "Perissodactyla", "Carnivora",
        "Rodentia", "Lagomorpha", "Chiroptera", "Mammalia", "Aves"}

_report = []


def log(message=""):
    print(message, flush=True)
    _report.append(str(message))


# Ranks at or below genus. A clade name shared with a genus (Ctenophora, Diplura) means the
# higher taxon, never the genus.
GENUS_AND_BELOW = {"genus", "subgenus", "species group", "species subgroup", "species",
                   "subspecies", "varietas", "forma", "strain", "isolate"}


def resolve_labels(ncbi):
    """
    Turn the clade name sets into taxid -> label.

    Stops the run if a name matches no taxon, or more than one above genus rank.
    """
    labels = {}
    for group, names in [("aquatic", AQUATIC), ("mangrove", MANGROVE),
                         ("terrestrial", TERRESTRIAL), ("host", HOST)]:
        for name in sorted(names):
            candidates = ncbi.get_name_translator([name]).get(name, [])
            ranks = ncbi.get_rank(candidates)
            if len(candidates) > 1:
                candidates = [t for t in candidates if ranks[t] not in GENUS_AND_BELOW]
            if len(candidates) != 1:
                raise SystemExit(f"clade {name!r}: expected one NCBI taxon, "
                                 f"found {candidates}")
            labels[candidates[0]] = group
    return labels


def classify_taxa(ncbi, taxids, labels):
    """
    Label each taxon by the first labelled clade on its lineage, walking towards the root.

    Returns
    -------
    dict
        taxid -> "aquatic", "mangrove", "terrestrial", "host" or "unassigned", for the
        taxids that have a lineage.
    """
    out = {}
    missing = []
    for taxid in taxids:
        with warnings.catch_warnings():
            # A retired taxid is followed through merged.dmp; ete4 warns each time.
            warnings.simplefilter("ignore")
            try:
                lineage = ncbi.get_lineage(taxid)
            except ValueError:
                missing.append(taxid)
                continue
        out[taxid] = "unassigned"
        for node in reversed(lineage):
            if node in labels:
                out[taxid] = labels[node]
                break
    log(f"  taxa with a lineage: {len(out):,} of {len(taxids):,} "
        f"({len(missing):,} not found)")
    return out


def resolve_names(ncbi, names):
    """
    Name -> taxid by c07's rules, from c07's cache where it has the name.

    Names missing from the cache are resolved here and not written back, so the cache
    stays the one c07 builds.
    """
    mapping = read_name_cache()
    missing = sorted(set(names) - set(mapping))
    if missing:
        mapping.update(resolve_to_taxids(ncbi, missing))
    out = {name: mapping[name] for name in names if name in mapping}
    return out, len(names) - len(out)


def load_collections():
    """COCONUT's collections field per InChIKey, pooled across entries sharing a key."""
    parents = pd.read_parquet(
        RESULTS / "coconut_parents.parquet", columns=["identifier", "inchikey"]
    )
    raw = pd.read_csv(COCONUT_CSV, usecols=["identifier", "collections"], dtype=str,
                      on_bad_lines="skip", low_memory=False)
    linked = parents.merge(raw, on="identifier").dropna(subset=["collections"])
    return linked.groupby("inchikey")["collections"].agg("|".join).reset_index()


def main():
    RESULTS.mkdir(exist_ok=True)
    log("WHICH SOURCE ORGANISMS LIVE IN WATER")
    log("=" * 78)
    if not COCONUT_CSV.exists():
        raise SystemExit(f"missing {COCONUT_CSV}; run c00_download_inputs.py")

    ncbi = cs.open_taxonomy()
    log(f"  NCBI taxonomy: {cs.TAXDUMP_ARCHIVE}")
    labels = resolve_labels(ncbi)

    coconut = load_organism_field().rename("organisms").reset_index()
    coconut = coconut.merge(load_collections(), on="inchikey", how="left")
    log(f"  molecules with an organism annotation: {len(coconut):,}")

    parsed = [split_organisms(v) for v in coconut.organisms]
    vocabulary = sorted({n for names in parsed for n in names})
    log(f"  distinct organism names: {len(vocabulary):,}")

    name2id, misses = resolve_names(ncbi, vocabulary)
    log(f"  resolved to an NCBI taxon: {len(name2id):,}  ({misses:,} unresolved)")

    taxon_label = classify_taxa(ncbi, sorted(set(name2id.values())), labels)
    name2label = {n: taxon_label[t] for n, t in name2id.items() if t in taxon_label}
    log("\n  organism names by habitat")
    for group, n in Counter(name2label.values()).most_common():
        log(f"    {group:<12} {n:6,}")

    rows, aquatic_seen = [], Counter()
    for inchikey, collections_field, names in zip(coconut.inchikey, coconut.collections,
                                                  parsed):
        counts = Counter(name2label.get(n, "unresolved") for n in names)
        aq, terr = counts["aquatic"], counts["terrestrial"]
        for n in names:
            if name2label.get(n) == "aquatic":
                aquatic_seen[n] += 1
        tokens = {u.strip() for t in str(collections_field).split("|") for u in t.split(";")}
        rows.append(dict(
            inchikey=inchikey, n_organisms=len(names),
            n_aquatic=aq, n_terrestrial=terr, n_mangrove=counts["mangrove"],
            n_host=counts["host"], n_unassigned=counts["unassigned"] + counts["unresolved"],
            share_aquatic=(aq / (aq + terr)) if (aq + terr) else np.nan,
            any_aquatic=aq > 0,
            marine_db=bool(tokens & MARINE_COLLECTIONS)))
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "aquatic_sources.csv", index=False)

    # The full table (one row per COCONUT molecule) is on Zenodo, not in Git. This copy holds
    # only the panel odorants, so Supplementary_Material/aquatic_origin runs on a fresh clone.
    if PANEL_JOIN.exists():
        keys = pd.read_csv(PANEL_JOIN, usecols=["inchikey"]).inchikey.dropna().unique()
        panel = table[table.inchikey.isin(keys)]
        panel.to_csv(RESULTS / "aquatic_sources_panel.csv", index=False)
        log(f"  aquatic_sources_panel.csv: {len(panel):,} of {len(table):,} rows")

    usable = table.share_aquatic.notna()
    log(f"\n  molecules written                  : {len(table):,}")
    log(f"  with at least one placed organism  : {usable.sum():,}")
    log(f"  with at least one aquatic organism : {int(table.any_aquatic.sum()):,}")
    log(f"  median share_aquatic where defined : {table.share_aquatic[usable].median():.4f}")
    log(f"  in a marine natural-product database: {int(table.marine_db.sum()):,}")

    seen = pd.DataFrame(aquatic_seen.most_common(), columns=["organism", "n_molecules"])
    seen["taxid"] = seen.organism.map(name2id)
    seen.to_csv(RESULTS / "aquatic_taxa_seen.csv", index=False)
    log("\n  most frequent aquatic source organisms")
    for _, r in seen.head(20).iterrows():
        log(f"    {r.n_molecules:5,}  {r.organism}")

    (RESULTS / "aquatic_sources_report.txt").write_text("\n".join(_report) + "\n")


if __name__ == "__main__":
    main()
