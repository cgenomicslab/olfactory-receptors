"""Step 12 -- does a molecule come from organisms that live in water?

Step 07 asked which kingdoms odorants are isolated from and step 10 asked it of the M2OR list.
This asks a different question of the same COCONUT organism field: of the organisms a
molecule was isolated from, how many live in water. The answer is written per molecule so
the receptor notebooks can join it to whatever grouping they are testing -- the immediate
consumer is section 12 of `Downstream_Analysis/notebooks/reference_tree/02.OR_human_landscape`,
which asks whether Class I receptors read more of the aquatic chemistry than Class II.

Habitat is not a field in COCONUT, so it has to come from the organism's position in the
NCBI tree. Three rules, applied to each organism's lineage from the tip upwards, first
match winning -- which is what makes the plant case work: a seagrass hits `Zosteraceae`
before it hits `Embryophyta`, and a whale hits `Cetacea` before `Artiodactyla`.

  aquatic      clades whose members live submerged or floating, in the sea or in fresh
               water. Macroalgae and the aquatic protist lineages; the animal phyla that
               are entirely aquatic; fish, crustaceans and the marine mammals; and -- the
               part that matters most here, because COCONUT is mostly plants -- the land
               plants that grow in water, from seagrasses to duckweed to water lilies.
  terrestrial  land plants, insects, arachnids and land vertebrates.
  host         mammals and birds, which turn up as "source organisms" when a compound was
               found in breath, milk or body odour rather than made by the animal. 07
               takes the same view. Excluded from the comparison rather than counted as
               terrestrial, because that is a statement about where a measurement was
               taken, not where the chemistry comes from.
  unassigned   everything else. Bacteria and fungi are the bulk of it: a Streptomyces is
               as likely to be from marine sediment as from soil and COCONUT does not say,
               so guessing would be worse than abstaining.

A molecule's `share_aquatic` is aquatic / (aquatic + terrestrial). Host and unassigned
organisms are in neither, so a molecule known only from bacteria has no share at all and
is reported as NaN rather than as zero.

Mangroves are deliberately *not* aquatic here. They are land trees rooted in the
intertidal, and calling them aquatic would put Rhizophora next to Zostera; they are
counted separately so the choice is visible and reversible.

Outputs
-------
results/aquatic_sources.csv       one row per COCONUT molecule with an organism annotation
results/aquatic_taxa_seen.csv     which aquatic taxa actually appear, and in how many
results/aquatic_sources_report.txt

Needs data/nodes.dmp and data/names.dmp from the NCBI taxonomy dump:

    cd data
    curl -O https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
    tar xzf taxdump.tar.gz nodes.dmp names.dmp

Parsed directly rather than through ete, so this step needs no taxonomy database built
into the user's home directory and no network at run time. CPU only, about two minutes.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"
NODES = DATA / "nodes.dmp"
NAMES = DATA / "names.dmp"
CACHE = RESULTS / "taxid_cache.json"          # 07's name -> taxid cache, reused if present

# COCONUT collections that are marine by construction. Used only as an independent check
# on the taxonomy: it is a property of the molecule's literature record, not of the
# occurrence, so it never feeds share_aquatic.
MARINE_COLLECTIONS = {"CMNPD", "Marine Natural Products",
                      "Seaweed Metabolite Database (SWMD)"}

# Clades that live in water. Resolved to taxids by name at run time, so there are no
# magic numbers here to rot when NCBI renumbers something.
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
    "Crustacea", "Cetacea", "Sirenia", "Pinnipedia", "Cheloniidae", "Dermochelyidae",
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

# Detected in, not made by. 07's convention, kept identical.
HOST = {"Homo sapiens", "Primates", "Artiodactyla", "Perissodactyla", "Carnivora",
        "Rodentia", "Lagomorpha", "Chiroptera", "Mammalia", "Aves"}

_report = []


def log(message=""):
    print(message, flush=True)
    _report.append(str(message))


def clean_organism_name(name):
    """Reduce one raw organism string to something NCBI might recognise (as in step 07)."""
    cleaned = re.sub(r"[^A-Za-z .\-]", " ", name)
    return re.sub(r"\s+", " ", cleaned).strip()


def split_organisms(field):
    """Split one organisms cell into its individual cleaned names (as in step 07)."""
    return [c for part in re.split(r"[|+]", str(field))
            for c in [clean_organism_name(part)] if len(c) > 2]


def load_taxonomy():
    """parent map, rank map and name->taxid map, straight out of the dump files."""
    parent, rank = {}, {}
    with open(NODES, encoding="latin-1") as handle:
        for line in handle:
            f = line.split("\t|\t")
            taxid, par = int(f[0]), int(f[1])
            parent[taxid] = par
            rank[taxid] = f[2]
    name2taxid, taxid2name = {}, {}
    with open(NAMES, encoding="latin-1") as handle:
        for line in handle:
            f = line.split("\t|\t")
            taxid, name, kind = int(f[0]), f[1], f[3].split("\t|")[0]
            if kind == "scientific name":
                taxid2name[taxid] = name
            name2taxid.setdefault(name, taxid)
    return parent, rank, name2taxid, taxid2name


def resolve_labels(name2taxid):
    """Turn the clade name sets into taxid -> label, warning about anything unresolved."""
    labels = {}
    for group, names in [("aquatic", AQUATIC), ("mangrove", MANGROVE),
                         ("terrestrial", TERRESTRIAL), ("host", HOST)]:
        for name in names:
            taxid = name2taxid.get(name)
            if taxid is None:
                log(f"  warning: no NCBI taxon called {name!r} — rule ignored")
                continue
            labels[taxid] = group
    return labels


def classify(taxid, parent, labels, memo):
    """Walk from the organism towards the root; the first labelled ancestor wins."""
    if taxid in memo:
        return memo[taxid]
    chain, node, out = [], taxid, "unassigned"
    seen = set()
    while node and node not in seen:
        seen.add(node)
        chain.append(node)
        if node in memo:
            out = memo[node]
            break
        if node in labels:
            out = labels[node]
            break
        nxt = parent.get(node)
        if nxt is None or nxt == node:
            break
        node = nxt
    for n in chain:
        memo[n] = out
    return out


def resolve_names(names, name2taxid):
    """Name -> taxid, falling back to the genus when the binomial is not in NCBI."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out, misses = {}, 0
    for name in names:
        taxid = cache.get(name) or name2taxid.get(name)
        if taxid is None:
            genus = name.split(" ")[0]
            taxid = cache.get(genus) or name2taxid.get(genus)
        if taxid is None:
            misses += 1
        else:
            out[name] = int(taxid) if not isinstance(taxid, list) else int(taxid[0])
    return out, misses


def main():
    RESULTS.mkdir(exist_ok=True)
    log("WHICH SOURCE ORGANISMS LIVE IN WATER")
    log("=" * 78)
    for path in (NODES, NAMES, COCONUT_CSV):
        if not path.exists():
            raise SystemExit(f"missing {path} — see the module docstring")

    parent, rank, name2taxid, taxid2name = load_taxonomy()
    log(f"  NCBI taxonomy: {len(parent):,} nodes")
    labels = resolve_labels(name2taxid)

    coconut = pd.read_csv(COCONUT_CSV, low_memory=False,
                          usecols=["standard_inchi_key", "organisms", "collections"])
    coconut = (coconut.rename(columns={"standard_inchi_key": "inchikey"})
               .dropna(subset=["organisms"]).drop_duplicates("inchikey"))
    log(f"  COCONUT molecules with an organism annotation: {len(coconut):,}")

    parsed = [split_organisms(v) for v in coconut.organisms]
    vocabulary = sorted({n for names in parsed for n in names})
    log(f"  distinct organism names: {len(vocabulary):,}")

    name2id, misses = resolve_names(vocabulary, name2taxid)
    log(f"  resolved to an NCBI taxon: {len(name2id):,}  ({misses:,} unresolved)")

    memo = {}
    name2label = {n: classify(t, parent, labels, memo) for n, t in name2id.items()}
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

    # A panel-sized copy, restricted to the molecules the odorant panel actually joins to.
    # The full table has one row per COCONUT molecule (~270k, ~15 MB) and stays out of Git
    # like the other large chemical-space outputs; this slice is a few hundred rows, so the
    # downstream notebook can run on a fresh clone without the 700 MB COCONUT download.
    panel_join = HERE / "panel_universe_join.csv"
    if panel_join.exists():
        keys = pd.read_csv(panel_join, usecols=["inchikey"]).inchikey.dropna().unique()
        panel = table[table.inchikey.isin(keys)]
        panel.to_csv(RESULTS / "aquatic_sources_panel.csv", index=False)
        print(f"  aquatic_sources_panel.csv: {len(panel):,} of {len(table):,} rows")

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
