"""Check whether this clone has everything it needs to run the analysis.

A fresh clone is not self-sufficient: the prediction matrices and the embeddings are
too large for Git and come from Zenodo instead, and the chemical-space pipeline needs a
700 MB download that is not redistributed here at all. This script says which of those
are present, which are missing, and the exact command that fetches each missing one --
so nobody discovers a gap three notebooks in.

Usage
-----
    python scripts/check_reproducibility.py            # everything
    python scripts/check_reproducibility.py --group downstream
    python scripts/check_reproducibility.py --quiet    # only what is missing

Exit code is 0 when every *required* item is present, 1 otherwise. Optional items
(the chemical-space pipeline, the per-residue embeddings) never affect the exit code,
because most people never need them.
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
    # A directory is checked for being non-empty rather than merely existing, because
    # an extracted-but-empty folder is the failure mode that actually happens.
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
COCONUT = "see Chemicals/README.md (700 MB COCONUT download)"

GROUPS = {
    "downstream": Group(
        "Downstream analysis",
        "the notebooks under Downstream_Analysis/ -- what most people came for",
        [
            Item("Downstream_Analysis/predictions/aggregated/ASR_Predictions/"
                 "asr_median_probability_wide.csv",
                 "432 ancestral nodes x 754 odorants, median probability", ZENODO),
            Item("Downstream_Analysis/predictions/aggregated/Reference_Tree_Predictions/"
                 "reference_median_probability_wide.csv",
                 "584 reference-tree proteins x 754 odorants", ZENODO),
            Item("Model_Inputs/Data_Preparation/processed/m2or_pairs_model.csv",
                 "measured M2OR pairs; the experimental layer and the SMILES map"),
            Item("Phylogenetic_Analysis/data/ReferenceTree/"
                 "PF13853.9606_7955_7740_7764_75743_137246.fa",
                 "receptor sequences, for the exact-sequence join"),
            Item("Phylogenetic_Analysis/data/HumanTree/human433_OR_classes.csv",
                 "Class I / Class II assignment per human receptor"),
            Item("Phylogenetic_Analysis/data/HumanTree/human433_MFP_reorder.tree",
                 "phylogenetic row order for the heatmaps in 02"),
            Item("Chemicals/processed_data/m2or_inchi_odors_smiles.csv",
                 "odour descriptors per molecule"),
            Item("Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/"
                 "PF13853_9606.7764.mafft.lg.treefile.rooted.withinternalnames",
                 "the rooted ASR tree with internal node names -- every ancestral "
                 "notebook walks it"),
        ],
    ),
    "chemspace": Group(
        "Chemical space (optional -- NEEDS A GPU)",
        "Chemicals/ -- odorants against a natural-product background. "
        "Step 02 requires a GPU and its output is not archived, so without one this "
        "analysis cannot be reproduced.",
        [
            Item("Chemicals/processed_data/odorants_m2or_leffingwell_goodscents.csv",
                 "the 5,962-molecule odorant list, release-independent"),
            Item("Chemicals/data/coconut_csv-08-2026.csv",
                 "the COCONUT natural-product dump", COCONUT, required=False),
            Item("Chemicals/results/universe.parquet",
                 "background + odorants with descriptors (built by c02)",
                 "python c02_build_universe.py", required=False),
            Item("Chemicals/results/embeddings.npy",
                 "MolFormer embeddings for the universe -- GPU only, not on Zenodo",
                 "python c03_embed.py   (needs a GPU)", required=False),
        ],
    ),
    "ancestral": Group(
        "Ancestral reconstruction runs (optional)",
        "the 433 per-node IQ-TREE runs behind the reconstructed sequences. Only needed "
        "to audit or re-derive them -- the sequences themselves are in the repository.",
        [
            Item("Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/"
                 "hagfish_human_ASR.final.fasta",
                 "the 432 reconstructed ancestral sequences"),
            Item("Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/"
                 "NODE_node_1",
                 "per-node IQ-TREE runs, 9.6 GB unpacked",
                 f"{ZENODO} --assets asr_node_runs", required=False, is_dir=True),
        ],
    ),
    "embeddings": Group(
        "Embeddings (optional)",
        "only needed to retrain the model or redo the embedding analyses",
        [
            Item("Model_Inputs/Embeddings/molecules/molformer_smiles_embeddings.pt",
                 "MolFormer ligand embeddings, 754 x 768", ZENODO, required=False),
            Item("Model_Inputs/Embeddings/proteins/reference_tree_esmc_300m_embeddings",
                 "per-residue ESM-C arrays, reference tree",
                 f"{ZENODO} --assets embeddings_reference_tree",
                 required=False, is_dir=True),
        ],
    ),
}

# Import name -> what breaks without it. Checked by import, not by version, because a
# working install is the only thing that actually matters here.
PACKAGES = [
    ("numpy", "everything"),
    ("pandas", "everything"),
    ("scipy", "statistics throughout"),
    ("sklearn", "the calibration map and every classifier"),
    ("matplotlib", "every figure"),
    ("seaborn", "most figures"),
    ("rdkit", "descriptors, functional groups, scaffolds"),
    ("ete4", "tree traversal -- all ancestral notebooks"),
    ("Bio", "FASTA parsing"),
    ("pyarrow", "the chemical-space .parquet files"),
    ("torch", "chemical-space kNN and matching"),
    ("thermo", "Joback boiling points -- SILENTLY returns NaN if missing"),
    ("umap", "chemical-space maps"),
    ("statsmodels", "some downstream statistics"),
    ("openpyxl", "the Supplementary Tables workbook"),
]

OPTIONAL_PACKAGES = {"thermo", "umap", "torch", "statsmodels", "openpyxl"}


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
            print("  conda env create -f environment.yml && conda activate olfactory-receptors")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--group", choices=sorted(GROUPS) + ["all"], default="all",
                        help="check only one group (default: all)")
    parser.add_argument("--quiet", action="store_true",
                        help="list only what is missing")
    arguments = parser.parse_args()

    selected = (list(GROUPS.values()) if arguments.group == "all"
                else [GROUPS[arguments.group]])

    print(f"Checking clone at {REPO}")

    missing_required, missing_optional = check_files(selected, arguments.quiet)
    missing_packages = bad_paths = 0
    if arguments.group == "all":
        missing_packages = check_packages(arguments.quiet)
        bad_paths = check_paths(arguments.quiet)

    print("\n" + "=" * 70)
    if missing_required or missing_packages or bad_paths:
        parts = []
        if missing_required:
            parts.append(f"{missing_required} required file(s)")
        if missing_packages:
            parts.append(f"{missing_packages} required package(s)")
        if bad_paths:
            parts.append(f"{bad_paths} file(s) with absolute paths")
        print("NOT READY: " + ", ".join(parts) + ".")
        print("Fix the lines marked [MISS] above, then run this again.")
    else:
        print("READY: every required input and package is present, "
              "and every path is relative.")
    if missing_optional:
        print(f"({missing_optional} optional item(s) absent -- fine unless you are "
              f"rebuilding that part.)")
    print("=" * 70)

    return 1 if (missing_required or missing_packages) else 0


if __name__ == "__main__":
    sys.exit(main())
