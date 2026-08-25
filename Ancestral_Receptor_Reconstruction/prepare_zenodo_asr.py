"""Create a versioned Zenodo archive of the per-node ancestral-reconstruction runs.

The 433 `NODE_node_*` directories hold everything IQ-TREE produced while reconstructing
each internal node: the filtered alignment it ran on, the posterior state file, the run
report, the tree with that node's branch lengths, and the extracted sequence and mask.
Together they are 9.6 GB, which is why they are not in Git.

They are archived rather than dropped because they cannot be regenerated without re-running
433 IQ-TREE jobs, and because the `.state` files are the only record of *how confident* each
reconstructed residue is. With them, the reconstruction can be re-derived under a different
rule -- sampling from the posterior instead of taking the maximum, say -- rather than being
taken on trust as a finished FASTA.

Packaged as-is: nothing is filtered out, so the archive is exactly what the pipeline wrote.

The ZIP unpacks directly into
`Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/`, recreating the
`NODE_node_*` tree where the pipeline expects it.

Usage:
  python Ancestral_Receptor_Reconstruction/prepare_zenodo_asr.py --version v1
  python Ancestral_Receptor_Reconstruction/prepare_zenodo_asr.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# The .state files are ~9.4 GB of the 9.6 and compress about 9x. Level 6 is the
# sweet spot: level 9 buys under 2% more for roughly triple the time.
COMPRESSLEVEL = 6

RUN_DIR = "human_tree_asr_hagfish_outgroup"


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


def node_files(run_root: Path) -> list[Path]:
    """Every file under every NODE_node_* directory, sorted by node number."""
    directories = sorted(
        run_root.glob("NODE_node_*"),
        key=lambda p: int(p.name.rsplit("_", 1)[1]),
    )
    if not directories:
        sys.exit(f"No NODE_node_* directories under {run_root}")
    return [path for directory in directories
            for path in sorted(directory.rglob("*")) if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1],
                        help="Project root. Default: auto-detected.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Ignored directory for release files. Default: <root>/release.")
    parser.add_argument("--version", default="v1", help="Dataset version label.")
    parser.add_argument("--archive-name", default=None,
                        help="Output ZIP name (default: olfactory_receptor_asr_node_runs_<version>.zip).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be packaged and exit.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing archive.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir
                  else project_root / "release")
    run_root = project_root / "Ancestral_Receptor_Reconstruction" / RUN_DIR

    files = node_files(run_root)
    raw_bytes = sum(path.stat().st_size for path in files)
    nodes = len({path.parent.name for path in files})
    print(f"{nodes} node directories, {len(files)} files, "
          f"{raw_bytes / 2**30:.2f} GiB raw")

    if args.dry_run:
        by_suffix: dict[str, list[int]] = {}
        for path in files:
            by_suffix.setdefault(path.suffix or path.name, []).append(path.stat().st_size)
        for suffix, sizes in sorted(by_suffix.items(), key=lambda kv: -sum(kv[1])):
            print(f"  {suffix:12s} {len(sizes):5d} files  {sum(sizes) / 2**20:9.0f} MiB")
        return

    name = args.archive_name or f"olfactory_receptor_asr_node_runs_{args.version}.zip"
    archive_path = output_dir / name
    if archive_path.exists() and not args.overwrite:
        sys.exit(f"Archive already exists: {archive_path}\nUse --overwrite to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_title": "Per-node ancestral-state reconstruction runs (IQ-TREE)",
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_node_directories": nodes,
        "n_files": len(files),
        "uncompressed_bytes": raw_bytes,
        "layout": "NODE_<node>/ one directory per internal node, exactly as the pipeline wrote it",
        "contents": {
            "human_sequences.<node>.fasta": "alignment restricted to the columns that node had",
            "human_sequences.<node>.fasta.state": "per-site posterior over all 20 amino acids",
            "human_sequences.<node>.fasta.iqtree": "IQ-TREE run report",
            "human_sequences.<node>.fasta.treefile": "tree with that run's branch lengths",
            "human_sequences.<node>.fasta.log": "console log",
            "human_sequences.<node>.fasta.ckp.gz": "IQ-TREE checkpoint",
            "<node>_ASR.final.fasta": "the reconstructed sequence",
            "<node>_mask.fasta": "presence/absence mask over the 337 alignment columns",
        },
        "note": (
            "Unpacks into Ancestral_Receptor_Reconstruction/human_tree_asr_hagfish_outgroup/. "
            "The reconstructed sequences are also published together in "
            "hagfish_human_ASR.final.fasta, which is in the Git repository; this archive is "
            "the evidence behind them."
        ),
    }

    with tempfile.TemporaryDirectory(dir=output_dir) as temp_dir:
        staging = Path(temp_dir)
        manifest_path = staging / "MANIFEST_asr_node_runs.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        temporary_archive = staging / archive_path.name
        with ZipFile(temporary_archive, "w", ZIP_DEFLATED, compresslevel=COMPRESSLEVEL) as archive:
            for position, path in enumerate(files, start=1):
                archive.write(path, str(path.relative_to(run_root)))
                if position % 250 == 0 or position == len(files):
                    print(f"  {position}/{len(files)}", flush=True)
            archive.write(manifest_path, "MANIFEST_asr_node_runs.json")
        shutil.move(temporary_archive, archive_path)

    checksum = write_checksum(archive_path)
    size = archive_path.stat().st_size
    print(f"\nCreated {archive_path.name} ({size / 2**20:.0f} MiB, "
          f"{raw_bytes / size:.1f}x compression)")
    print(f"  SHA-256 {checksum}")
    print(f"\nAdd to zenodo_manifest.json:")
    print(json.dumps({"asr_node_runs": {
        "filename": archive_path.name,
        "sha256": checksum,
        "destination": f"Ancestral_Receptor_Reconstruction/{RUN_DIR}",
        "bytes": size,
        "description": (f"Per-node IQ-TREE ancestral-state runs, {nodes} directories. "
                        "Large; only needed to re-derive or audit the reconstruction."),
    }}, indent=2))


if __name__ == "__main__":
    main()
