"""Step 1 -- build the universe: odorants plus a natural-product background.

The ChEMBL comparison showed odorants sitting apart from the rest of chemical space. But
ChEMBL is mostly drug chemistry, so a lot of that separation was really "odorants are not
drugs" -- odorants carry no nitrogen and plenty of oxygen, and drugs are the opposite.
COCONUT is a database of natural products, which are themselves nitrogen-poor and
oxygen-rich, so swapping the background to COCONUT takes that easy contrast away.

If odorants still stand apart from natural products, odour chemistry is genuinely its own
thing. If they do not, the honest conclusion is that odorants are simply the volatile
natural products, and the ChEMBL result was a gap between two curation styles.

This script does the bookkeeping for that comparison: read COCONUT, clean it up, add the
odorants, and describe every molecule.

Inputs
------
data/coconut_csv-08-2026.csv   the COCONUT dump (see README; not committed, it is ~700 MB)
data/reference_sets.csv        the 5,962 odorants (M2OR + Leffingwell + GoodScents)

Outputs
-------
results/coconut_parents.parquet   cleaned COCONUT; cached, so re-runs are cheap
results/universe.parquet          background + odorants, with descriptors for each

CPU only. About 30 minutes on 48 processes, or 3 on 64. Nearly all of that is the
functional-group matching and the boiling-point estimates over 738k molecules.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import chemspace as cs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"
COCONUT_CLEANED = RESULTS / "coconut_parents.parquet"

# The odorant list. Built from M2OR and pyrfume rather than from COCONUT, so it carries no
# dependency on which COCONUT release we happen to be using.
ODORANTS = DATA / "reference_sets.csv"

# Columns we read from the COCONUT dump. Reading only these keeps the 700 MB file manageable.
COCONUT_COLUMNS = [
    "identifier", "canonical_smiles", "np_likeness", "contains_sugar",
    "np_classifier_pathway", "np_classifier_superclass", "np_classifier_class",
    "np_classifier_is_glycoside", "chemical_super_class", "annotation_level",
]

# The subset of those that are annotations rather than structure. Odorants that also appear
# in COCONUT inherit these, so an odorant can carry a biosynthetic pathway too.
ANNOTATION_COLUMNS = [
    "identifier", "np_likeness", "contains_sugar", "np_classifier_pathway",
    "np_classifier_superclass", "np_classifier_class", "np_classifier_is_glycoside",
    "chemical_super_class", "annotation_level",
]

ODORANT_SOURCES = ["in_m2or", "in_leffingwell", "in_goodscents"]


def canonicalise(smiles):
    """Module-level wrapper so multiprocessing can pickle it."""
    return cs.to_parent(smiles)


def load_coconut(n_processes):
    """
    Read COCONUT and reduce every entry to a clean parent structure.

    Desalting and canonicalising 738k molecules takes a while, so the result is cached as
    a parquet file. After the first run this function just reads that back.

    Parameters
    ----------
    n_processes : int
        Worker processes for the desalting step.

    Returns
    -------
    DataFrame
        The COCONUT entries that parsed, with "smiles" and "inchikey" columns added.
    """
    if COCONUT_CLEANED.exists():
        print(f"using cached {COCONUT_CLEANED}", flush=True)
        return pd.read_parquet(COCONUT_CLEANED)

    if not COCONUT_CSV.exists():
        raise SystemExit(
            f"missing {COCONUT_CSV}\n"
            "  see the README: download and unzip the COCONUT dump first"
        )

    print("reading COCONUT csv ...", flush=True)
    coconut = pd.read_csv(
        COCONUT_CSV,
        usecols=COCONUT_COLUMNS,
        dtype={"identifier": str, "canonical_smiles": str},
        on_bad_lines="skip",
        low_memory=False,
    )
    coconut = coconut.dropna(subset=["canonical_smiles"])

    print(f"  {len(coconut)} rows; desalting on {n_processes} procs ...", flush=True)
    with Pool(n_processes) as pool:
        parents = pool.map(canonicalise, list(coconut.canonical_smiles), chunksize=400)

    parsed = [i for i, parent in enumerate(parents) if parent is not None]
    print(f"  parsed {len(parsed)} / {len(coconut)}", flush=True)

    coconut = coconut.iloc[parsed].reset_index(drop=True)
    coconut["smiles"] = [parents[i][0] for i in parsed]
    coconut["inchikey"] = [parents[i][1] for i in parsed]

    RESULTS.mkdir(exist_ok=True)
    coconut.to_parquet(COCONUT_CLEANED, index=False)
    print(f"  wrote {COCONUT_CLEANED}", flush=True)
    return coconut


def load_odorants(n_processes, coconut_keys):
    """
    Read the odorant list and canonicalise it the same way COCONUT was.

    Both sides have to go through `cs.to_parent`, or the InChIKeys will not line up and
    the overlap between the two will look far smaller than it is.

    Parameters
    ----------
    n_processes : int
        Worker processes.
    coconut_keys : set of str
        InChIKeys present in COCONUT, used only for the coverage report.

    Returns
    -------
    DataFrame
        One row per odorant, deduplicated on InChIKey.
    """
    odorants = pd.read_csv(ODORANTS)

    with Pool(n_processes) as pool:
        parents = pool.map(canonicalise, list(odorants.smiles_can), chunksize=50)

    parsed = [i for i, parent in enumerate(parents) if parent is not None]
    odorants = odorants.iloc[parsed].reset_index(drop=True)
    odorants["smiles"] = [parents[i][0] for i in parsed]
    odorants["inchikey"] = [parents[i][1] for i in parsed]
    odorants = odorants.drop_duplicates("inchikey")

    # How much of the odorant list COCONUT already knows about. Reported per source
    # because the three lists were curated differently and overlap COCONUT differently.
    n_in_coconut = odorants.inchikey.isin(coconut_keys).sum()
    print(
        f"\nodorants: {len(odorants)};  present in COCONUT: "
        f"{n_in_coconut} ({100 * n_in_coconut / len(odorants):.1f}%)",
        flush=True,
    )
    for source in ODORANT_SOURCES:
        from_source = odorants[odorants[source] == 1]
        overlap = from_source.inchikey.isin(coconut_keys).sum()
        print(
            f"    {source:16s} {overlap}/{len(from_source)} "
            f"({100 * overlap / len(from_source):.1f}%)",
            flush=True,
        )
    return odorants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-background", type=int, default=0,
                        help="how many natural products to keep; 0 means all of COCONUT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=48)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    coconut = load_coconut(args.nproc)
    coconut_unique = coconut.drop_duplicates("inchikey", keep="first").reset_index(drop=True)
    print(
        f"COCONUT: {len(coconut)} rows -> {len(coconut_unique)} unique parent structures",
        flush=True,
    )
    coconut_keys = set(coconut_unique.inchikey)

    odorants = load_odorants(args.nproc, coconut_keys)

    # Build the background. Odorants are taken out of the pool so that "background" means
    # natural products that are NOT odorants -- otherwise we would be comparing odorants
    # against a set that partly contains them.
    background = coconut_unique[~coconut_unique.inchikey.isin(set(odorants.inchikey))]
    if args.n_background and args.n_background < len(background):
        rng = np.random.default_rng(args.seed)
        chosen = np.sort(rng.choice(len(background), args.n_background, replace=False))
        background = background.iloc[chosen]
    background = background.copy()
    background["in_background"] = 1
    print(f"background: {len(background)} natural products", flush=True)

    # The odorant side of the universe, carrying COCONUT's annotations where COCONUT
    # happens to know the molecule.
    odorant_side = odorants[["smiles", "inchikey"] + ODORANT_SOURCES].copy()
    odorant_side["in_background"] = 0
    odorant_side = odorant_side.join(
        coconut_unique.set_index("inchikey")[ANNOTATION_COLUMNS], on="inchikey"
    )

    universe = pd.concat([background, odorant_side], ignore_index=True, sort=False)
    for source in ODORANT_SOURCES:
        universe[source] = universe[source].fillna(0).astype(int)

    # A molecule is an odorant if any of the three lists contains it.
    universe["is_odorant"] = (universe[ODORANT_SOURCES].sum(axis=1) > 0).astype(int)
    universe["in_coconut"] = universe.inchikey.isin(coconut_keys).astype(int)
    universe = universe.drop(columns=["canonical_smiles"], errors="ignore")
    assert universe.inchikey.is_unique, "InChIKey collision in the universe"

    print(
        f"\ncomputing descriptors + functional groups + boiling points for "
        f"{len(universe)} ...",
        flush=True,
    )
    with Pool(args.nproc) as pool:
        described = pool.map(cs.descriptors, list(universe.smiles), chunksize=200)

    usable = [i for i, values in enumerate(described) if values is not None]
    universe = universe.iloc[usable].reset_index(drop=True)
    universe = pd.concat(
        [universe, pd.DataFrame([described[i] for i in usable])], axis=1
    )

    is_odorant = universe.is_odorant == 1
    is_background = universe.in_background == 1

    print("\n### universe")
    print(f"  total                {len(universe)}")
    print(f"  natural products     {int(is_background.sum())}")
    print(
        f"  odorants             {int(is_odorant.sum())}   "
        f"(of which in COCONUT: {int((is_odorant & (universe.in_coconut == 1)).sum())})"
    )

    print("\n### medians, odorants vs natural products")
    for column in ["MolWt", "Tb_joback", "logP_vap", "LogP", "TPSA", "nN", "nO",
                   "AromaticRings", "NumStereo", "np_likeness"]:
        if column not in universe:
            continue
        odorant_median = pd.to_numeric(universe[column][is_odorant], errors="coerce").median()
        background_median = pd.to_numeric(
            universe[column][is_background], errors="coerce"
        ).median()
        print(
            f"  {column:16s} odorants={odorant_median:9.2f}   "
            f"natural products={background_median:9.2f}"
        )

    universe.to_parquet(RESULTS / "universe.parquet", index=False)
    print(f"\nwrote {RESULTS / 'universe.parquet'}  shape={universe.shape}")


if __name__ == "__main__":
    main()
