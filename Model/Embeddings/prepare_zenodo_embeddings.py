"""Create versioned Zenodo archives of the ESM-C 300M protein embeddings.

Two kinds of archive are produced, and they go into the same Zenodo record as
the downstream predictions:

  pooled      One small ZIP holding the mean-pooled (N, 960) matrix per dataset
              plus a row index. This is the representation the model consumes:
              the extract_embeddings_* notebooks reduce each per-residue matrix
              with emb.mean(axis=0), which is reproduced here exactly.

  per-residue One ZIP per dataset holding the raw (L, 960) float32 arrays.
              Split per dataset so nobody has to download 3 GB to get one set.
              These cannot be regenerated without a GPU and ESM-C, which is why
              they are worth archiving despite the size.

The pooled matrices are derived from the per-residue arrays, so publishing both
lets anyone verify the reduction rather than trust it.
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
from typing import Dict, List, Optional, Tuple
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


DATASETS = {
    "gpcr": "gpcr_esmc_300m_embeddings",
    "m2or": "m2or_esmc_300m_embeddings",
    "reference_tree": "reference_tree_esmc_300m_embeddings",
}

EMBEDDING_DIM = 960

# MolFormer ligand embeddings. The .pt is keyed by SMILES string, but the
# prediction matrices are keyed by SML id, so the archive also carries the
# id<->SMILES mapping and a torch-free .npy copy.
MOLECULE_FILE = "molformer_smiles_embeddings.pt"
MOLECULE_DIM = 768

# float32 embeddings deflate to ~93% of raw, so heavy compression buys almost
# nothing on the multi-GB per-residue archives but costs a lot of time.
PER_RESIDUE_COMPRESSLEVEL = 1
POOLED_COMPRESSLEVEL = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksum(archive_path: Path) -> str:
    checksum = sha256(archive_path)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{checksum}  {archive_path.name}\n", encoding="utf-8")
    return checksum


def dataset_files(source_root: Path, directory: str) -> List[Path]:
    dataset_dir = source_root / directory
    if not dataset_dir.is_dir():
        sys.exit(f"Embedding directory not found: {dataset_dir}")
    files = sorted(dataset_dir.glob("*.npy"))
    if not files:
        sys.exit(f"No .npy files under: {dataset_dir}")
    return files


def pool_dataset(files: List[Path]) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Mean-pool each per-residue array, exactly as the extract notebooks do."""
    matrix = np.empty((len(files), EMBEDDING_DIM), dtype=np.float32)
    index: List[Dict[str, object]] = []
    for row, path in enumerate(files):
        array = np.load(path)
        if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
            sys.exit(f"Unexpected embedding shape {array.shape} in {path}")
        matrix[row] = array.mean(axis=0)
        index.append({"row": row, "id": path.stem, "n_residues": array.shape[0], "file": path.name})
    return matrix, index


def build_pooled_archive(
    source_root: Path,
    output_dir: Path,
    selected: List[str],
    version: str,
    overwrite: bool,
) -> Optional[Path]:
    archive_path = output_dir / f"olfactory_receptor_protein_embeddings_pooled_{version}.zip"
    if archive_path.exists() and not overwrite:
        sys.exit(f"Archive already exists: {archive_path}\nUse --overwrite to replace it.")

    entries = []
    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
        staging = Path(temp_dir_name)
        for name in selected:
            files = dataset_files(source_root, DATASETS[name])
            print(f"Pooling {name}: {len(files)} proteins")
            matrix, index = pool_dataset(files)

            matrix_path = staging / f"{name}_esmc300m_pooled.npy"
            np.save(matrix_path, matrix)
            index_path = staging / f"{name}_esmc300m_index.csv"
            with index_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["row", "id", "n_residues", "file"])
                writer.writeheader()
                writer.writerows(index)
            print(f"  -> {matrix.shape} float32, {matrix_path.stat().st_size / 2**20:.1f} MiB")
            entries += [matrix_path, index_path]

        manifest = {
            "dataset_title": "Olfactory receptor protein embeddings (ESM-C 300M, mean-pooled)",
            "version": version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model": "ESM-C 300M",
            "embedding_dim": EMBEDDING_DIM,
            "pooling": "mean over the residue axis of the per-residue (L, 960) array",
            "datasets": selected,
            "note": (
                "Row order matches the accompanying *_index.csv. These matrices are "
                "derived from the per-residue archives in this same record."
            ),
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in entries
            ],
        }
        manifest_path = staging / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        temporary_archive = staging / archive_path.name
        with ZipFile(temporary_archive, "w", ZIP_DEFLATED, compresslevel=POOLED_COMPRESSLEVEL) as archive:
            for path in entries:
                archive.write(path, f"pooled/{path.name}")
            archive.write(manifest_path, "MANIFEST.json")
        shutil.move(temporary_archive, archive_path)

    checksum = write_checksum(archive_path)
    print(f"Created {archive_path.name} ({archive_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"  SHA-256 {checksum}\n")
    return archive_path


def build_per_residue_archive(
    source_root: Path,
    output_dir: Path,
    name: str,
    version: str,
    overwrite: bool,
) -> Path:
    archive_path = output_dir / f"olfactory_receptor_protein_embeddings_{name}_{version}.zip"
    if archive_path.exists() and not overwrite:
        sys.exit(f"Archive already exists: {archive_path}\nUse --overwrite to replace it.")

    directory = DATASETS[name]
    files = dataset_files(source_root, directory)
    raw_bytes = sum(path.stat().st_size for path in files)
    print(f"Packaging {name}: {len(files)} arrays, {raw_bytes / 2**30:.2f} GiB raw")

    manifest = {
        "dataset_title": f"Olfactory receptor protein embeddings (ESM-C 300M, per-residue, {name})",
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": "ESM-C 300M",
        "embedding_dim": EMBEDDING_DIM,
        "layout": f"{directory}/<id>.npy, each a float32 (L, {EMBEDDING_DIM}) array",
        "n_files": len(files),
        "uncompressed_bytes": raw_bytes,
        "note": "Mean-pooling these over the residue axis reproduces the pooled matrices in this record.",
    }

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
        staging = Path(temp_dir_name)
        manifest_path = staging / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        temporary_archive = staging / archive_path.name
        with ZipFile(temporary_archive, "w", ZIP_DEFLATED, compresslevel=PER_RESIDUE_COMPRESSLEVEL) as archive:
            for position, path in enumerate(files, start=1):
                archive.write(path, f"{directory}/{path.name}")
                if position % 250 == 0 or position == len(files):
                    print(f"  {position}/{len(files)}")
            archive.write(manifest_path, "MANIFEST.json")
        shutil.move(temporary_archive, archive_path)

    checksum = write_checksum(archive_path)
    print(f"Created {archive_path.name} ({archive_path.stat().st_size / 2**30:.2f} GiB)")
    print(f"  SHA-256 {checksum}\n")
    return archive_path


def build_molecule_archive(
    molecule_path: Path,
    pairs_csv: Path,
    output_dir: Path,
    version: str,
    overwrite: bool,
) -> Path:
    """Package the MolFormer ligand embeddings with their SML id mapping."""
    import pandas as pd
    import torch

    archive_path = output_dir / f"olfactory_receptor_molecule_embeddings_{version}.zip"
    if archive_path.exists() and not overwrite:
        sys.exit(f"Archive already exists: {archive_path}\nUse --overwrite to replace it.")
    if not molecule_path.is_file():
        sys.exit(f"Molecule embeddings not found: {molecule_path}")
    if not pairs_csv.is_file():
        sys.exit(f"SMILES id mapping not found: {pairs_csv}")

    embeddings = torch.load(molecule_path, map_location="cpu", weights_only=False)
    if not isinstance(embeddings, dict):
        sys.exit(f"Expected a dict of SMILES -> tensor in {molecule_path}")

    mapping = (
        pd.read_csv(pairs_csv, usecols=["SMILES", "smiles_id"])
        .drop_duplicates()
        .sort_values("smiles_id")
        .reset_index(drop=True)
    )
    missing = [s for s in mapping["SMILES"] if s not in embeddings]
    if missing:
        sys.exit(f"{len(missing)} SMILES in {pairs_csv.name} have no embedding, e.g. {missing[:3]}")
    unmapped = len(embeddings) - len(mapping)
    if unmapped:
        print(f"  ! {unmapped} embedded SMILES have no SML id and are omitted from the index")

    print(f"Packaging molecules: {len(mapping)} ligands")
    matrix = np.empty((len(mapping), MOLECULE_DIM), dtype=np.float32)
    for row, smiles in enumerate(mapping["SMILES"]):
        matrix[row] = embeddings[smiles].detach().cpu().numpy().astype(np.float32)

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir_name:
        staging = Path(temp_dir_name)
        matrix_path = staging / "molformer_molecule_embeddings.npy"
        np.save(matrix_path, matrix)
        index_path = staging / "molformer_molecule_index.csv"
        mapping.assign(row=range(len(mapping)))[["row", "smiles_id", "SMILES"]].to_csv(
            index_path, index=False
        )
        original_copy = staging / MOLECULE_FILE
        shutil.copy2(molecule_path, original_copy)
        entries = [matrix_path, index_path, original_copy]
        print(f"  -> {matrix.shape} float32, index keyed by smiles_id")

        manifest = {
            "dataset_title": "Olfactory receptor ligand embeddings (MolFormer)",
            "version": version,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model": "MolFormer",
            "embedding_dim": MOLECULE_DIM,
            "n_ligands": len(mapping),
            "note": (
                "molformer_molecule_embeddings.npy rows follow molformer_molecule_index.csv, "
                "whose smiles_id values match the ligand columns of the prediction matrices. "
                f"{MOLECULE_FILE} is the original torch dict keyed by SMILES string."
            ),
            "files": [
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in entries
            ],
        }
        manifest_path = staging / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        temporary_archive = staging / archive_path.name
        with ZipFile(temporary_archive, "w", ZIP_DEFLATED, compresslevel=POOLED_COMPRESSLEVEL) as archive:
            for path in entries:
                archive.write(path, f"molecules/{path.name}")
            archive.write(manifest_path, "molecules/MANIFEST.json")
        shutil.move(temporary_archive, archive_path)

    checksum = write_checksum(archive_path)
    print(f"Created {archive_path.name} ({archive_path.stat().st_size / 2**20:.1f} MiB)")
    print(f"  SHA-256 {checksum}\n")
    return archive_path


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "Model" / "Embeddings" / "proteins",
        help="Directory holding the *_esmc_300m_embeddings folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "release",
        help="Ignored directory in which to write release files.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help=f"Comma-separated subset of {','.join(DATASETS)}, or 'all'.",
    )
    parser.add_argument("--version", default="v1", help="Dataset version label.")
    parser.add_argument(
        "--pooled-only",
        action="store_true",
        help="Build only the small pooled and molecule archives, skipping the multi-GB per-residue ones.",
    )
    parser.add_argument(
        "--molecule-file",
        type=Path,
        default=project_root / "Model" / "Embeddings" / "molecules" / MOLECULE_FILE,
        help="MolFormer ligand embeddings (torch dict keyed by SMILES).",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=project_root / "Model" / "Data_Preparation" / "processed" / "m2or_pairs_model.csv",
        help="Source of the smiles_id <-> SMILES mapping.",
    )
    parser.add_argument(
        "--skip-molecules",
        action="store_true",
        help="Do not build the MolFormer ligand embedding archive.",
    )
    parser.add_argument(
        "--molecules-only",
        action="store_true",
        help=(
            "Build only the MolFormer ligand archive. Useful for adding it without "
            "rebuilding the others, which would change their checksums."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing archives.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets == "all":
        selected = list(DATASETS)
    else:
        selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
        unknown = [name for name in selected if name not in DATASETS]
        if unknown:
            sys.exit(f"Unknown dataset(s): {', '.join(unknown)}. Choose from {', '.join(DATASETS)}.")

    print(f"Source  : {source_root}")
    print(f"Output  : {output_dir}")
    print(f"Datasets: {', '.join(selected)}\n")

    if args.molecules_only:
        build_molecule_archive(
            args.molecule_file.resolve(),
            args.pairs_csv.resolve(),
            output_dir,
            args.version,
            args.overwrite,
        )
        return

    build_pooled_archive(source_root, output_dir, selected, args.version, args.overwrite)

    if not args.skip_molecules:
        build_molecule_archive(
            args.molecule_file.resolve(),
            args.pairs_csv.resolve(),
            output_dir,
            args.version,
            args.overwrite,
        )

    if args.pooled_only:
        print("Skipping per-residue archives (--pooled-only).")
        return
    for name in selected:
        build_per_residue_archive(source_root, output_dir, name, args.version, args.overwrite)

    print("Upload every .zip and .sha256 in the output directory to the Zenodo record.")


if __name__ == "__main__":
    main()
