# Aggregate model probabilities across multiple runs for each available category
# category (ASR_Predictions, M2OR_Unseen_Pairs_Predictions, and/or
# Reference_Tree_Predictions).

# Input:
# Model-run CSV files with columns including protein_id, smiles_id, and probability.

# Processing:
# 1. find the folder with prediction files
# 2. choose one or all categories
# 3. read the 5 run files of a category
# 4. stack them together in one long table
# 5. group by (protein_id, smiles_id)
# 6. compute summary statistics across runs
# 7. reshape median probabilities into a wide matrix
# 8. save the probability matrix

# Outputs:
# For each category, the script exports:
# - median probability wide matrix
# Binary calls are deliberately not written here. Apply the analysis-specific
# threshold in the relevant notebook; figure2_activation_code (Part 1) derives the
# calibrated value (0.915) from the measured pairs.

# Usage examples:
# python Downstream_Analysis/scripts/aggregate_prediction_runs.py --dry-run
# python Downstream_Analysis/scripts/aggregate_prediction_runs.py
# python Downstream_Analysis/scripts/aggregate_prediction_runs.py --category ASR_Predictions


from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


CATEGORY_CONFIG = {
    "ASR_Predictions": {
        "file_regex": r"ASR_predictions_run_(\d+)\.csv$",
        "prefix": "asr",
    },
    "M2OR_Unseen_Pairs_Predictions": {
        "file_regex": r"unseen_pairs_predictions_run_(\d+)\.csv$",
        "prefix": "unseen",
    },
    "Reference_Tree_Predictions": {
        "file_regex": r"reference_tree_predictions_run_(\d+)\.csv$",
        "prefix": "reference",
    },
}


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root_default = script_path.parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate model-run probabilities by exact (protein_id, smiles_id) "
            "pair and export median-probability matrices."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root_default,
        help="Project root directory. Default: auto-detected from script location.",
    )

    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help=(
            "Explicit path to a Predictions_M2OR-* archive directory. "
            "If omitted, the latest one under Downstream_Analysis/predictions/archive/ "
            "will be selected automatically."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Output root directory. Default: "
            "<project-root>/Downstream_Analysis/predictions/aggregated"
        ),
    )

    parser.add_argument(
        "--category",
        choices=list(CATEGORY_CONFIG.keys()) + ["all"],
        default="all",
        help="Process only one category or all categories.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing output files.",
    )

    parser.add_argument(
        "--limit-runs",
        type=int,
        default=None,
        help=(
            "For quick testing only: process only the first N runs found per category. "
            "Normally leave unset."
        ),
    )

    return parser.parse_args()


def detect_latest_archive_dir(archive_root: Path) -> Path:
    """
    Detect the latest Predictions_M2OR-* directory by sorting names lexicographically.
    This works well with timestamped directory names like:
    Predictions_M2OR-20251005T081853Z-1-001
    """
    candidates = [
        p for p in archive_root.iterdir()
        if p.is_dir() and p.name.startswith("Predictions_M2OR-")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No Predictions_M2OR-* directories found under: {archive_root}"
        )

    latest = sorted(candidates, key=lambda p: p.name)[-1]
    return latest


def has_any_category(data_root: Path) -> bool:
    """True if the directory holds at least one configured category subdirectory."""
    return any((data_root / category).is_dir() for category in CATEGORY_CONFIG)


def resolve_prediction_data_root(project_root: Path, archive_dir: Optional[Path]) -> Path:
    """
    Locate the directory holding the per-category run files, trying in order:

    1. <archive-dir>/Predictions_M2OR/All_Data_No_Mixtures_Predictions, when
       --archive-dir is given explicitly.
    2. Downstream_Analysis/predictions/runs, the layout produced by the Zenodo
       release and unpacked by scripts/download_zenodo_data.py.
    3. The latest Predictions_M2OR-* export under predictions/archive/, which is
       the raw model output as exported locally.
    """
    if archive_dir is not None:
        data_root = archive_dir.resolve() / "Predictions_M2OR" / "All_Data_No_Mixtures_Predictions"
        if not data_root.is_dir():
            raise FileNotFoundError(f"Expected data root not found: {data_root}")
        return data_root

    predictions_root = project_root / "Downstream_Analysis" / "predictions"

    runs_root = predictions_root / "runs"
    if has_any_category(runs_root):
        return runs_root

    archive_root = predictions_root / "archive"
    if archive_root.is_dir():
        data_root = (
            detect_latest_archive_dir(archive_root)
            / "Predictions_M2OR"
            / "All_Data_No_Mixtures_Predictions"
        )
        if has_any_category(data_root):
            return data_root

    raise FileNotFoundError(
        f"No prediction run files found under {runs_root} or {archive_root}.\n"
        "Download the published dataset first:\n"
        "  python scripts/download_zenodo_data.py"
    )


def find_run_files(category_dir: Path, pattern: str, category_name: str, limit_runs: Optional[int] = None) -> List[Path]:
    regex = re.compile(pattern)
    matched = []

    for fp in category_dir.iterdir():
        if fp.is_file():
            m = regex.search(fp.name)
            if m:
                run_num = int(m.group(1))
                matched.append((run_num, fp))

    matched.sort(key=lambda x: x[0])
    run_files = [fp for _, fp in matched]

    if not run_files:
        raise ValueError(f"No run files found for category: {category_name}")

    if limit_runs is not None:
        if limit_runs < 1:
            raise ValueError("--limit-runs must be >= 1")
        run_files = run_files[:limit_runs]

    return run_files


def load_and_combine_runs(run_files: List[Path]) -> pd.DataFrame:
    expected_columns = {"protein_id", "smiles_id", "probability"}
    dfs = []

    for fp in run_files:
        df = pd.read_csv(fp)

        missing = expected_columns - set(df.columns)
        if missing:
            raise ValueError(f"{fp} is missing required columns: {sorted(missing)}")

        run_match = re.search(r"run_(\d+)\.csv$", fp.name)
        if run_match is None:
            raise ValueError(f"Could not extract run number from filename: {fp.name}")

        run_num = int(run_match.group(1))

        df = df[["protein_id", "smiles_id", "probability"]].copy()
        df["run"] = run_num
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    return combined


def aggregate_runs(combined_df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        combined_df
        .groupby(["protein_id", "smiles_id"], as_index=False)
        .agg(
            n_runs=("run", "nunique"),
            mean_probability=("probability", "mean"),
            median_probability=("probability", "median"),
            std_probability=("probability", "std"),
        )
    )

    return agg_df


def make_wide_matrix(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = df.pivot(index="protein_id", columns="smiles_id", values=value_col)
    wide = wide.sort_index(axis=0).sort_index(axis=1)
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def save_csv(df: pd.DataFrame, path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[DRY-RUN] Would write: {path}")
        print(f"[DRY-RUN] Shape: {df.shape}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote: {path}")


def process_category(
    category_name: str,
    config: Dict[str, str],
    data_root: Path,
    output_root: Path,
    dry_run: bool = False,
    limit_runs: Optional[int] = None,
) -> None:
    category_dir = data_root / category_name
    if not category_dir.exists():
        raise FileNotFoundError(f"Category directory not found: {category_dir}")

    run_files = find_run_files(category_dir, config["file_regex"], category_name, limit_runs=limit_runs)

    print(f"\nProcessing category: {category_name}")
    for fp in run_files:
        print(f"  - {fp.name}")

    combined_df = load_and_combine_runs(run_files)

    agg_df = aggregate_runs(combined_df)
    out_dir = output_root / category_name
    prefix = config["prefix"]

    median_prob_wide = make_wide_matrix(agg_df, "median_probability")
    median_prob_path = out_dir / f"{prefix}_median_probability_wide.csv"
    save_csv(median_prob_wide, median_prob_path, dry_run=dry_run)


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_root / "Downstream_Analysis" / "predictions" / "aggregated"
    )

    data_root = resolve_prediction_data_root(project_root, args.archive_dir)

    print(f"Project root : {project_root}")
    print(f"Data root    : {data_root}")
    print(f"Output root  : {output_root}")
    print(f"Dry run      : {args.dry_run}")
    print(f"Limit runs   : {args.limit_runs}")

    if args.category == "all":
        categories = [
            category_name
            for category_name in CATEGORY_CONFIG
            if (data_root / category_name).is_dir()
        ]
        missing = set(CATEGORY_CONFIG) - set(categories)
        if missing:
            print(f"Skipping unavailable categories: {', '.join(sorted(missing))}")
        if not categories:
            raise FileNotFoundError(f"No configured prediction categories found under: {data_root}")
    else:
        categories = [args.category]

    for category_name in categories:
        process_category(
            category_name=category_name,
            config=CATEGORY_CONFIG[category_name],
            data_root=data_root,
            output_root=output_root,
            dry_run=args.dry_run,
            limit_runs=args.limit_runs,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
