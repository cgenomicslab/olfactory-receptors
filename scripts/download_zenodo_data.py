"""Download and verify published data assets from the project's Zenodo record.

One Zenodo record holds several archives: the downstream prediction
probabilities and the ESM-C protein embeddings. Each is listed in
zenodo_manifest.json with its own checksum and extraction destination.

By default only the small, immediately useful assets are fetched (predictions
and the pooled embeddings). The multi-GB per-residue embedding archives are
opt-in.

Usage:
  python scripts/download_zenodo_data.py                     # default assets
  python scripts/download_zenodo_data.py --list              # show what exists
  python scripts/download_zenodo_data.py --assets all
  python scripts/download_zenodo_data.py --assets embeddings_gpcr
  python scripts/download_zenodo_data.py --base-url https://sandbox.zenodo.org
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List
from zipfile import ZipFile


DEFAULT_BASE_URL = "https://zenodo.org"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe path in archive: {member.filename}")
    archive.extractall(destination)


def report_progress(name: str, downloaded: int, total: int) -> None:
    if total <= 0:
        sys.stdout.write(f"\r  {name}: {downloaded / 2**20:.1f} MiB")
    else:
        share = downloaded / total
        sys.stdout.write(
            f"\r  {name}: {downloaded / 2**20:>8.1f} / {total / 2**20:.1f} MiB ({share:5.1%})"
        )
    sys.stdout.flush()


def download_asset(
    key: str,
    asset: Dict[str, object],
    record_id: str,
    base_url: str,
    project_root: Path,
    force: bool,
) -> None:
    filename = asset["filename"]
    expected = asset.get("sha256")
    if not expected:
        sys.exit(
            f"Asset '{key}' has no sha256 in the manifest. Fill it in from the "
            f"generated .sha256 file before downloading."
        )

    destination = (project_root / asset["destination"]).resolve()
    if destination.exists() and any(destination.iterdir()) and not force:
        sys.exit(
            f"Destination for '{key}' already contains files: {destination}\n"
            "Use --force only if you intend to merge into it."
        )

    url = f"{base_url.rstrip('/')}/records/{record_id}/files/{filename}?download=1"
    print(f"\n{key}: {url}")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir_name:
        downloaded_path = Path(temp_dir_name) / filename
        with urllib.request.urlopen(url) as response, downloaded_path.open("wb") as output:
            total = int(response.headers.get("Content-Length", 0))
            seen = 0
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                seen += len(chunk)
                report_progress(key, seen, total)
        print()

        observed = sha256(downloaded_path)
        if observed.lower() != expected.lower():
            sys.exit(f"  Checksum mismatch for {filename}.\n  expected {expected}\n  got      {observed}")
        print(f"  checksum OK, extracting into {destination}")
        with ZipFile(downloaded_path) as archive:
            safe_extract(archive, destination)


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "zenodo_manifest.json",
        help="Zenodo release metadata JSON.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Root that asset destinations are resolved against.",
    )
    parser.add_argument(
        "--assets",
        default=None,
        help="Comma-separated asset keys, or 'all'. Default: the manifest's default_assets.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Zenodo instance. Use https://sandbox.zenodo.org to rehearse. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument("--list", action="store_true", help="List available assets and exit.")
    parser.add_argument("--force", action="store_true", help="Allow extraction into a non-empty destination.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    assets: Dict[str, Dict[str, object]] = manifest["assets"]

    if args.list:
        print(f"{manifest['record_title']} ({manifest['version']})")
        print(f"record id: {manifest.get('zenodo_record_id') or 'NOT PUBLISHED YET'}\n")
        for key, asset in assets.items():
            size = asset.get("bytes")
            size_text = f"{size / 2**20:.1f} MiB" if size else "size unknown"
            ready = "ready" if asset.get("sha256") else "NO CHECKSUM"
            print(f"  {key:<26} {size_text:>12}  [{ready}]  -> {asset['destination']}")
            print(f"  {'':<26} {asset.get('description', '')}")
        print(f"\ndefault: {', '.join(manifest.get('default_assets', []))}")
        return

    record_id = manifest.get("zenodo_record_id")
    if not record_id:
        sys.exit(
            "zenodo_record_id is not set in the manifest.\n"
            "Publish the Zenodo record first, then record its numeric id."
        )

    if args.assets == "all":
        selected: List[str] = list(assets)
    elif args.assets:
        selected = [key.strip() for key in args.assets.split(",") if key.strip()]
    else:
        selected = list(manifest.get("default_assets", assets))

    unknown = [key for key in selected if key not in assets]
    if unknown:
        sys.exit(f"Unknown asset(s): {', '.join(unknown)}. Known: {', '.join(assets)}")

    for key in selected:
        download_asset(key, assets[key], record_id, args.base_url, args.project_root.resolve(), args.force)

    print(f"\nDone. Downloaded and verified: {', '.join(selected)}")


if __name__ == "__main__":
    main()
