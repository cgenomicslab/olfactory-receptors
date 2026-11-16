"""Build the chemical-space Zenodo archives into release/.

Each asset is keyed to its entry in zenodo_manifest.json, and each archive keeps the
internal prefix its manifest destination expects, so unzipping it there recreates the
layout the code reads. The other archives are built by

    Downstream_Analysis/scripts/prepare_zenodo_predictions.py
    Model_Inputs/Embeddings/prepare_zenodo_embeddings.py
    Ancestral_Receptor_Reconstruction/prepare_zenodo_asr.py

Usage:
  python scripts/prepare_zenodo_assets.py --list
  python scripts/prepare_zenodo_assets.py --chemicals --version v3 --update-manifest
  python scripts/prepare_zenodo_assets.py --chemical-embeddings --version v1 --update-manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "zenodo_manifest.json"
RELEASE_DIR = PROJECT_ROOT / "release"


# Each entry: (path relative to the project root, path inside the archive).
# The archive prefix must match what the manifest destination expects.
ASSET_SPECS: Dict[str, Dict[str, object]] = {
    "chemical_space": {
        "group": "chemicals",
        "summary": "Odorants against a natural-product background, COCONUT 08-2026.",
        "required": [
            ("Chemicals/results/universe.parquet", "results/universe.parquet"),
            ("Chemicals/results/coconut_parents.parquet", "results/coconut_parents.parquet"),
            ("Chemicals/results/embed_index.csv", "results/embed_index.csv"),
            ("Chemicals/results/umap.npy", "results/umap.npy"),
        ],
        "optional": [
            ("Chemicals/results/aquatic_sources.csv", "results/aquatic_sources.csv"),
            ("Chemicals/results/aquatic_taxa_seen.csv", "results/aquatic_taxa_seen.csv"),
        ],
    },
    "chemical_space_embeddings": {
        "group": "chemical-embeddings",
        "summary": "MoLFormer embedding of the chemical universe, rows in embed_index.csv order.",
        "required": [
            ("Chemicals/results/embeddings.npy", "results/embeddings.npy"),
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(spec: Dict[str, object]) -> List[tuple]:
    """Resolve a spec to (source, archive_name) pairs, failing on missing required files."""
    members: List[tuple] = []
    missing: List[str] = []

    for rel, arcname in spec.get("required", []):
        source = PROJECT_ROOT / rel
        if source.is_file():
            members.append((source, arcname))
        else:
            missing.append(rel)

    if missing:
        sys.exit("Missing required file(s):\n  " + "\n  ".join(missing))

    for rel, arcname in spec.get("optional", []):
        source = PROJECT_ROOT / rel
        if source.is_file():
            members.append((source, arcname))
        else:
            print(f"  optional, not present: {rel}")

    return members


def build(key: str, version: str, dry_run: bool) -> Optional[Path]:
    spec = ASSET_SPECS[key]
    members = collect(spec)
    total = sum(source.stat().st_size for source, _ in members)

    out_name = f"olfactory_receptor_{key}_{version}.zip"
    out_path = RELEASE_DIR / out_name

    print(f"\n{key} -> {out_name}")
    for source, arcname in members:
        print(f"  {source.stat().st_size / 2**20:>9.1f} MiB  {arcname}")
    print(f"  {'':>9}       ---")
    print(f"  {total / 2**20:>9.1f} MiB  uncompressed, {len(members)} files")

    if dry_run:
        print("  dry run, nothing written")
        return None

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    inner = {
        "asset": key,
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": spec.get("summary", ""),
        "notes": spec.get("notes", ""),
        "files": [],
    }

    temp_path = out_path.with_suffix(".zip.partial")
    with ZipFile(temp_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source, arcname in members:
            print(f"  writing {arcname} ...", flush=True)
            archive.write(source, arcname)
            inner["files"].append(
                {"name": arcname, "bytes": source.stat().st_size, "sha256": sha256(source)}
            )
        archive.writestr("MANIFEST.json", json.dumps(inner, indent=2) + "\n")

    temp_path.replace(out_path)

    checksum = sha256(out_path)
    (out_path.with_suffix(".zip.sha256")).write_text(f"{checksum}  {out_name}\n", encoding="utf-8")

    size = out_path.stat().st_size
    print(f"\n  {out_path}")
    print(f"  {size / 2**20:.1f} MiB compressed ({size / total:.0%} of {total / 2**20:.0f} MiB)")
    print(f"  sha256 {checksum}")
    return out_path


def update_manifest(key: str, archive: Path) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    asset = manifest["assets"][key]
    asset["filename"] = archive.name
    asset["sha256"] = sha256(archive)
    asset["bytes"] = archive.stat().st_size
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  manifest updated: assets.{key}.sha256 and .bytes")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    groups = {spec["group"]: key for key, spec in ASSET_SPECS.items()}

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for group, key in groups.items():
        parser.add_argument(f"--{group}", action="store_true", help=f"build {key}")
    parser.add_argument("--version", default=manifest.get("version", "v1"), help="Archive version tag.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be packaged, write nothing.")
    parser.add_argument("--update-manifest", action="store_true", help="Write checksum and size back to the manifest.")
    parser.add_argument("--list", action="store_true", help="Show the assets this script can build.")
    args = parser.parse_args()

    if args.list:
        print(f"builds into {RELEASE_DIR}\n")
        for group, key in groups.items():
            spec = ASSET_SPECS[key]
            n = len(spec.get("required", [])) + len(spec.get("optional", []))
            print(f"  --{group:<12} {key:<18} up to {n} files")
            print(f"  {'':<14} {spec.get('summary', '')}")
        return

    selected = [key for group, key in groups.items() if getattr(args, group.replace("-", "_"), False)]
    if not selected:
        parser.error("nothing selected. Use --list to see the options.")

    for key in selected:
        archive = build(key, args.version, args.dry_run)
        if archive and args.update_manifest:
            update_manifest(key, archive)


if __name__ == "__main__":
    main()
