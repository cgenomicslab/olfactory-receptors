"""Check whether this clone has what it needs to redraw the figures and rerun the analyses.

    python scripts/check_reproducibility.py            # everything
    python scripts/check_reproducibility.py --quiet    # only what is missing
    python scripts/check_reproducibility.py --zenodo   # also check the Zenodo record (internet)

Exit code 0 (READY) only when every figure can be redrawn and every analysis rerun: every
required file and package is present and no path is absolute. Optional items are only needed
to rerun the steps upstream of the predictions.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

TICK, CROSS, DASH = "OK  ", "MISS", "--  "


@dataclass
class Item:
    """One thing the analysis needs, and what to do when it is absent."""

    path: str
    purpose: str
    fix: str = ""
    required: bool = True
    # A directory counts as present only if it is not empty, so an archive that was
    # extracted into an empty folder is reported as missing.
    is_dir: bool = False

    def present(self) -> bool:
        target = REPO / self.path
        if self.is_dir:
            return target.is_dir() and any(target.iterdir())
        return target.is_file()


@dataclass
class Group:
    name: str
    blurb: str
    items: list = field(default_factory=list)


ZENODO = "python scripts/download_zenodo_data.py"
CHEMICALS = f"{ZENODO} --chemicals"
DRAWINGS = "Phylogenetic_Analysis/figures/"
SIX = "PF13853.9606_7955_7740_7764_75743_137246"

GROUPS = {
    "figures": Group(
        "Redraw the figures",
        "the tables the figures are drawn from, the chemical-space map and the tree drawings",
        [
            Item("Downstream_Analysis/extant_receptors/results", "Fig. 2, Table 1, Ext. Data 2-4",
                 "run Downstream_Analysis/extant_receptors/e01-e05", is_dir=True),
            Item("Downstream_Analysis/ancestral_receptors/results", "Fig. 3, Ext. Data 5-6",
                 "run Downstream_Analysis/ancestral_receptors/a01-a07", is_dir=True),
            Item("Chemicals/results/fg_enrichment.csv", "Fig. 4d"),
            Item("Chemicals/results/universe.parquet", "Fig. 4a, Ext. Data 7", CHEMICALS),
            Item("Chemicals/results/umap.npy", "Fig. 4a, Ext. Data 7", CHEMICALS),
            Item("Chemicals/results/embed_index.csv", "Fig. 4a, Ext. Data 7", CHEMICALS),
            Item(DRAWINGS + "hGPCR.1.svg", "Fig. 1a"),
            Item(DRAWINGS + "six_species.itol.svg", "Fig. 1c"),
            Item(DRAWINGS + "six_species_clade_support.svg", "Fig. 1d"),
            Item(DRAWINGS + "human433_MFP_reorder.tree.itol.svg", "Ext. Data 3"),
            Item(DRAWINGS + "human_hagfish_434.itol.svg", "Fig. 3a"),
        ],
    ),
    "analysis": Group(
        "Rerun the analyses",
        "the notebooks under Downstream_Analysis/",
        [
            Item("Downstream_Analysis/predictions/aggregated/ASR_Predictions/asr_median_probability_wide.csv",
                 "432 ancestral nodes x 754 odorants", ZENODO),
            Item("Downstream_Analysis/predictions/aggregated/Reference_Tree_Predictions/"
                 "reference_median_probability_wide.csv", "584 reference-tree receptors x 754 odorants", ZENODO),
            Item("Model_Inputs/Data_Preparation/processed/m2or_pairs_model.csv", "the measured pairs"),
            Item(f"Phylogenetic_Analysis/data/ReferenceTree/{SIX}.fa", "receptor sequences, for the exact-sequence join"),
            Item("Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv", "class I / class II"),
            Item("Phylogenetic_Analysis/data/HumanTree/human433_MFP_reorder.tree", "tree order, Ext. Data 3"),
            Item("Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/"
                 "PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames", "the ancestral tree"),
            Item("Chemicals/processed_data/m2or_inchi_odors_smiles.csv", "odour descriptors per odorant"),
            Item("Chemicals/processed_data/ligand_pathways.csv", "biosynthetic pathway per odorant"),
        ],
    ),
    "upstream": Group(
        "Rerun the steps upstream of the predictions (optional)",
        "embeddings, the chemical-space pipeline and the per-node ancestral runs",
        [
            Item("Model_Inputs/Embeddings/proteins/gpcr_esmc_300m_embeddings",
                 "ESM-C arrays of the 722 class A GPCRs, for the Fig. 1 embedding maps", f"{ZENODO} --embeddings",
                 required=False, is_dir=True),
            Item("Model_Inputs/Embeddings/proteins/reference_tree_esmc_300m_embeddings",
                 "ESM-C arrays of the 584 six-species receptors, for the Fig. 1 embedding maps", f"{ZENODO} --embeddings",
                 required=False, is_dir=True),
            Item("Model_Inputs/Embeddings/proteins/m2or_esmc_300m_embeddings",
                 "ESM-C arrays of the 1,399 M2OR receptors", f"{ZENODO} --embeddings-full",
                 required=False, is_dir=True),
            Item("Chemicals/results/embeddings.npy", "MoLFormer embedding of the universe, for c04-c06",
                 f"{ZENODO} --chemical-embeddings", required=False),
            Item("Chemicals/data/coconut_csv-08-2026.csv", "COCONUT, for c02, c07, c08 and c10",
                 "cd Chemicals && python c00_download_inputs.py", required=False),
            Item("Chemicals/data/taxa.sqlite", "NCBI taxonomy of 2026-08-01, for c07, c08, c10",
                 "cd Chemicals && python c00_download_inputs.py", required=False),
            Item("Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/NODE_node_1",
                 "the per-node IQ-TREE runs", f"{ZENODO} --asr", required=False, is_dir=True),
        ],
    ),
}

# Import name -> what needs it.
PACKAGES = [
    ("numpy", "everything"),
    ("pandas", "everything"),
    ("scipy", "statistics"),
    ("sklearn", "calibration, PCA, matching"),
    ("matplotlib", "every figure"),
    ("rdkit", "descriptors and functional groups"),
    ("ete4", "trees"),
    ("Bio", "FASTA parsing"),
    ("pyarrow", "the chemical-space tables"),
    ("umap", "the embedding maps"),
    ("torch", "the chemical-space kNN and matching"),
    ("thermo", "Joback boiling points; without it they are NaN"),
    ("openpyxl", "the Supplementary Tables workbook"),
    ("nbconvert", "running the notebooks"),
]

OPTIONAL_PACKAGES = set()


def check_files(groups, quiet=False):
    """Report on every file item. Returns (n_missing_required, n_missing_optional)."""
    missing_required = missing_optional = 0

    for group in groups:
        shown = False
        for item in group.items:
            ok = item.present()
            if ok and quiet:
                continue
            if not shown:
                print(f"\n{group.name}")
                print(f"  {group.blurb}")
                shown = True
            if ok:
                mark = TICK
            elif item.required:
                mark = CROSS
                missing_required += 1
            else:
                mark = DASH
                missing_optional += 1
            print(f"  [{mark}] {item.path}")
            if not ok:
                print(f"         {item.purpose}")
                if item.fix:
                    print(f"         get it with: {item.fix}")
    return missing_required, missing_optional


def check_packages(quiet=False):
    """Report on the Python environment. Returns the number of required imports missing."""
    missing = 0
    header_shown = False
    for name, purpose in PACKAGES:
        try:
            importlib.import_module(name)
            if quiet:
                continue
            mark, note = TICK, ""
        except Exception:
            optional = name in OPTIONAL_PACKAGES
            mark = DASH if optional else CROSS
            note = f"  <- {purpose}"
            if not optional:
                missing += 1
        if not header_shown:
            print("\nPython environment")
            print("  conda env create -f olfaction-env.yaml && conda activate olfactory-receptors")
            header_shown = True
        print(f"  [{mark}] {name}{note}")
    return missing


# Absolute paths that would only exist on the machine they were written on. Anything
# matching this cannot work from a fresh clone, which is the whole point of the layout:
# Zenodo archives extract into the exact directories the code already reads from, so
# every path in the codebase stays relative to the repository root.
ABSOLUTE_PATH = re.compile(
    r'["\']((?:/(?:data|home|Users|mnt|media|scratch|nfs)[^"\'\n]{3,})'
    r'|(?:[A-Za-z]:\\\\[^"\'\n]+))["\']'
)


def check_paths(quiet=False):
    """Scan code and notebooks for hard-coded absolute paths.

    Returns the number of files containing one. Comment lines are skipped, so a path
    mentioned in a docstring or a commented-out alternative does not count.
    """
    offenders = {}

    def scan(path, text):
        for number, line in enumerate(text.split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            for match in ABSOLUTE_PATH.finditer(line):
                offenders.setdefault(path, []).append((number, match.group(1)[:80]))

    for path in REPO.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        scan(path.relative_to(REPO), path.read_text(errors="ignore"))

    for path in REPO.rglob("*.ipynb"):
        if ".git" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        try:
            notebook = json.loads(path.read_text(errors="ignore"))
        except Exception:
            continue
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                scan(path.relative_to(REPO), "".join(cell["source"]))

    if offenders or not quiet:
        print("\nPortability")
        print("  every path in the codebase must be relative to the repository root")
    for path, found in sorted(offenders.items()):
        print(f"  [{CROSS}] {path}")
        for number, text in found[:3]:
            print(f"         L{number}: {text}")
    if not offenders and not quiet:
        print(f"  [{TICK}] no absolute paths in any .py or .ipynb")
    return len(offenders)


def check_zenodo(quiet=False):
    """Confirm every archive in zenodo_manifest.json is on the published record.

    Reads the record's file list from the Zenodo API and compares each manifest entry by
    file name and size. Returns the number of archives that could not be found, or 1 if
    the record itself could not be reached.
    """
    import urllib.request

    manifest = json.loads((REPO / "zenodo_manifest.json").read_text(encoding="utf-8"))
    record_id = manifest.get("zenodo_record_id")
    print("\nZenodo record")
    if not record_id:
        print(f"  [{CROSS}] zenodo_record_id is not set in zenodo_manifest.json")
        return 1
    url = f"https://zenodo.org/api/records/{record_id}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            record = json.load(response)
    except Exception as error:
        print(f"  [{CROSS}] could not read {url}: {error}")
        return 1

    published = {f["key"]: f["size"] for f in record.get("files", [])}
    missing = 0
    for key, asset in manifest["assets"].items():
        name, size = asset["filename"], asset.get("bytes")
        if name not in published:
            print(f"  [{CROSS}] {key}: {name} is not on record {record_id}")
            missing += 1
        elif size and published[name] != size:
            print(f"  [{CROSS}] {key}: {name} is {published[name]} bytes on Zenodo, "
                  f"{size} in the manifest")
            missing += 1
        elif not quiet:
            print(f"  [{TICK}] {key}: {name}")
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--group", choices=sorted(GROUPS) + ["all"], default="all",
                        help="check only one group (default: all)")
    parser.add_argument("--quiet", action="store_true",
                        help="list only what is missing")
    parser.add_argument("--zenodo", action="store_true",
                        help="also check every manifest archive is on the Zenodo record "
                             "(needs internet)")
    arguments = parser.parse_args()

    selected = (list(GROUPS.values()) if arguments.group == "all"
                else [GROUPS[arguments.group]])

    print(f"Checking clone at {REPO}")

    missing_required, missing_optional = check_files(selected, arguments.quiet)
    missing_packages = bad_paths = missing_archives = 0
    if arguments.group == "all":
        missing_packages = check_packages(arguments.quiet)
        bad_paths = check_paths(arguments.quiet)
    if arguments.zenodo:
        missing_archives = check_zenodo(arguments.quiet)

    problems = []
    if missing_required:
        problems.append(f"{missing_required} required file(s)")
    if missing_packages:
        problems.append(f"{missing_packages} required package(s)")
    if bad_paths:
        problems.append(f"{bad_paths} file(s) with absolute paths")
    if missing_archives:
        problems.append(f"{missing_archives} manifest archive(s) not downloadable from Zenodo")

    print("\n" + "=" * 70)
    if problems:
        print("NOT READY: " + ", ".join(problems) + ".")
    else:
        checked = "every figure can be redrawn and every analysis rerun"
        if arguments.zenodo:
            checked += "; every archive is on Zenodo"
        print(f"READY: {checked}.")
    if missing_optional:
        print(f"{missing_optional} optional item(s) absent: needed only for the upstream steps.")
    print("=" * 70)

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
