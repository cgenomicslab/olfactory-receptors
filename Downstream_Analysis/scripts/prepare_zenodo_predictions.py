"""Create a versioned Zenodo archive of the downstream prediction dataset.

The archive carries probabilities only. Binary calls are deliberately absent:
each analysis notebook applies whichever threshold it is testing, so a
pre-thresholded table would silently fix a choice that belongs downstream.

Contents:
  runs/<Category>/<run files>          per-run probabilities, all five runs
  aggregated/<Category>/<matrix>       median-probability wide matrices
  model_run_evaluation_metrics.csv     per-run evaluation metrics
  MANIFEST.json                        per-file sizes and SHA-256 checksums

The ZIP unpacks directly into Downstream_Analysis/predictions/. Both the
predictions and release directories are intentionally ignored by Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile


# Categories to publish. M2OR unseen pairs are deliberately excluded: they are
# not used by the downstream analysis and would triple the archive size.
RELEASE_CATEGORIES = {
    "ASR_Predictions": {
        "run_glob": "ASR_predictions_run_*.csv",
        "median_matrix": "asr_median_probability_wide.csv",
    },
    "Reference_Tree_Predictions": {
        "run_glob": "reference_tree_predictions_run_*.csv",
        "median_matrix": "reference_median_probability_wide.csv",
    },
}

# Columns kept from each per-run file. The source files also carry a
# `prediction` column thresholded at 0.5, which is dropped here.
RUN_COLUMNS = ["protein_id", "smiles_id", "probability"]

METRICS_FILENAME = "Runs_Model_Evaluation_Metrics.csv"
METRICS_RELEASE_NAME = "model_run_evaluation_metrics.csv"

EXPECTED_RUNS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_latest_archive_dir(archive_root: Path) -> Path:
    """Pick the newest Predictions_M2OR-* export by lexicographic name order."""
    candidates = [
        path for path in archive_root.iterdir()
        if path.is_dir() and path.name.startswith("Predictions_M2OR-")
    ]
    if not candidates:
        sys.exit(f"No Predictions_M2OR-* directories found under: {archive_root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def resolve_run_root(project_root: Path, archive_dir: Optional[Path]) -> Path:
    """Return the directory holding the per-category run subdirectories."""
    archive_root = project_root / "Downstream_Analysis" / "predictions" / "archive"
    if archive_dir is None:
        archive_dir = detect_latest_archive_dir(archive_root)
    else:
        archive_dir = archive_dir.resolve()

    run_root = archive_dir / "Predictions_M2OR" / "All_Data_No_Mixtures_Predictions"
    if not run_root.is_dir():
        sys.exit(f"Expected run directory not found: {run_root}")
    return run_root


def strip_prediction_column(source: Path, destination: Path) -> int:
    """Copy a run file keeping only RUN_COLUMNS. Returns the row count.

    Values are copied as text rather than parsed and re-serialised, so the
    published probabilities are byte-identical to the model output.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", newline="", encoding="utf-8") as handle_in:
        reader = csv.reader(handle_in)
        try:
            header = next(reader)
        except StopIteration:
            sys.exit(f"Run file is empty: {source}")

        missing = [column for column in RUN_COLUMNS if column not in header]
        if missing:
            sys.exit(f"{source} is missing required columns: {', '.join(missing)}")
        keep = [header.index(column) for column in RUN_COLUMNS]

        rows = 0
        with destination.open("w", newline="", encoding="utf-8") as handle_out:
            writer = csv.writer(handle_out, lineterminator="\n")
            writer.writerow(RUN_COLUMNS)
            for row in reader:
                writer.writerow([row[index] for index in keep])
                rows += 1
    return rows


def collect_run_files(run_root: Path, category: str, run_glob: str) -> List[Path]:
    category_dir = run_root / category
    if not category_dir.is_dir():
        sys.exit(f"Category directory not found: {category_dir}")
    run_files = sorted(category_dir.glob(run_glob))
    if not run_files:
        sys.exit(f"No run files matching {run_glob} under: {category_dir}")
    if len(run_files) != EXPECTED_RUNS:
        print(f"  ! {category}: found {len(run_files)} runs, expected {EXPECTED_RUNS}")
    return run_files


def stage_release(
    run_root: Path,
    aggregated_root: Path,
    staging: Path,
    include_medians: bool,
) -> Iterator[Tuple[Path, Path]]:
    """Materialise the release tree under `staging`, yielding (path, arcname)."""
    for category, config in RELEASE_CATEGORIES.items():
        print(f"Packaging {category}")
        for source in collect_run_files(run_root, category, config["run_glob"]):
            relative = Path("runs") / category / source.name
            target = staging / relative
            rows = strip_prediction_column(source, target)
            saved = source.stat().st_size - target.stat().st_size
            print(f"  - {source.name}: {rows} rows, dropped {saved / 1024 / 1024:.1f} MiB")
            yield target, relative

        if not include_medians:
            continue
        median_source = aggregated_root / category / config["median_matrix"]
        if not median_source.is_file():
            sys.exit(
                f"Median matrix not found: {median_source}\n"
                "Regenerate it with aggregate_prediction_runs.py, or pass --no-medians."
            )
        relative = Path("aggregated") / category / median_source.name
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(median_source, target)
        print(f"  - {median_source.name}: median matrix")
        yield target, relative

    metrics_source = run_root / METRICS_FILENAME
    if metrics_source.is_file():
        relative = Path(METRICS_RELEASE_NAME)
        target = staging / relative
        shutil.copy2(metrics_source, target)
        print(f"Packaging {METRICS_RELEASE_NAME}")
        yield target, relative
    else:
        print(f"! Evaluation metrics not found, skipping: {metrics_source}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root directory. Default: auto-detected from script location.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help=(
            "Explicit Predictions_M2OR-* export directory holding the per-run files. "
            "Default: the latest one under Downstream_Analysis/predictions/archive/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Ignored directory in which to write release files. Default: Downstream_Analysis/release.",
    )
    parser.add_argument("--version", default="v1", help="Dataset version label.")
    parser.add_argument(
        "--archive-name",
        default=None,
        help="Output ZIP name (default: olfactory_receptor_downstream_probabilities_<version>.zip).",
    )
    parser.add_argument(
        "--no-medians",
        action="store_true",
        help="Publish per-run probabilities only, without the derived median matrices.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing release ZIP.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "Downstream_Analysis" / "release"
    )
    aggregated_root = project_root / "Downstream_Analysis" / "predictions" / "aggregated"
    run_root = resolve_run_root(project_root, args.archive_dir)

    archive_name = args.archive_name or f"olfactory_receptor_downstream_probabilities_{args.version}.zip"
    archive_path = output_dir / archive_name
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    if archive_path.exists() and not args.overwrite:
        sys.exit(f"Release archive already exists: {archive_path}\nUse --overwrite to replace it.")

    print(f"Run source   : {run_root}")
    print(f"Aggregated   : {aggregated_root}")
    print(f"Output       : {archive_path}")
    print(f"Medians      : {'excluded' if args.no_medians else 'included'}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        staging = temp_dir / "staging"
        staging.mkdir()

        files = list(stage_release(run_root, aggregated_root, staging, not args.no_medians))

        manifest = {
            "dataset_title": "Olfactory receptor downstream prediction probabilities",
            "version": args.version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "categories": sorted(RELEASE_CATEGORIES),
            "source_export": run_root.parents[1].name,
            "contents": (
                "Per-run and median probabilities for the ASR and reference-tree "
                "predictions. Probabilities only: no binary call tables are "
                "distributed, since each analysis applies its own threshold."
            ),
            "layout": "Extract into Downstream_Analysis/predictions/.",
            "files": [
                {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path, relative in files
            ],
        }
        manifest_path = staging / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        temporary_archive = temp_dir / archive_name
        with ZipFile(temporary_archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for path, relative in files:
                archive.write(path, relative.as_posix())
            archive.write(manifest_path, "MANIFEST.json")
        shutil.move(temporary_archive, archive_path)

    archive_checksum = sha256(archive_path)
    checksum_path.write_text(f"{archive_checksum}  {archive_path.name}\n", encoding="utf-8")

    total_bytes = sum(entry["bytes"] for entry in manifest["files"])
    print(f"\nCreated : {archive_path} ({archive_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    print(f"Checksum: {checksum_path}")
    print(f"Files   : {len(files) + 1} ({total_bytes / 1024 / 1024:.1f} MiB uncompressed)")
    print(f"SHA-256 : {archive_checksum}")
    print("\nUpload both files to Zenodo. After publication, copy the record ID and")
    print("this SHA-256 into zenodo_manifest.json at the repository root.")


if __name__ == "__main__":
    main()
