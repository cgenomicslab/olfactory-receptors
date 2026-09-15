"""Download and verify published data assets from the project's Zenodo record.

One Zenodo record holds every archive: the prediction probabilities, the ESM-C
protein embeddings, the MolFormer molecule embeddings, the per-node ancestral
reconstruction runs and the chemical-space tables. Each is listed in
zenodo_manifest.json with its own checksum and extraction destination.

Assets are selected by group. Every group in the manifest becomes a flag here,
so adding one is a manifest edit rather than a code change.

Usage:
  python scripts/download_zenodo_data.py                     # default assets, ~49 MiB
  python scripts/download_zenodo_data.py --list              # show groups and assets
  python scripts/download_zenodo_data.py --predictions --chemicals
  python scripts/download_zenodo_data.py --asr
  python scripts/download_zenodo_data.py --embeddings-full   # includes the per-residue sets
  python scripts/download_zenodo_data.py --all
  python scripts/download_zenodo_data.py --assets embeddings_gpcr   # one archive by key
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
from typing import Dict, List, Tuple
from zipfile import ZipFile


DEFAULT_BASE_URL = "https://zenodo.org"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "zenodo_manifest.json"


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
            f"Asset '{key}' has no sha256 in the manifest. Build the archive and "
            f"fill in its checksum before downloading."
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


def load_manifest(argv: List[str]) -> Tuple[Path, Dict[str, object]]:
    """Read the manifest before building the real parser, since its groups define the flags."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    known, _ = pre.parse_known_args(argv)
    path = known.manifest.resolve()
    if not path.is_file():
        sys.exit(f"Manifest not found: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def build_parser(groups: Dict[str, List[str]]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Zenodo release metadata JSON.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root that asset destinations are resolved against.",
    )

    selection = parser.add_argument_group(
        "what to download",
        "Combine freely. With none of these, the manifest's default_assets are fetched.",
    )
    for name, keys in groups.items():
        selection.add_argument(
            f"--{name}",
            action="store_true",
            help=f"{', '.join(keys)}",
        )
    selection.add_argument("--all", action="store_true", help="Every asset in the manifest.")
    selection.add_argument(
        "--assets",
        default=None,
        help="Comma-separated asset keys, for selecting a single archive by name.",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Zenodo instance. Use https://sandbox.zenodo.org to rehearse. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument("--list", action="store_true", help="List groups and assets, then exit.")
    parser.add_argument("--force", action="store_true", help="Allow extraction into a non-empty destination.")
    return parser


def describe(manifest: Dict[str, object]) -> None:
    assets: Dict[str, Dict[str, object]] = manifest["assets"]
    groups: Dict[str, List[str]] = manifest.get("groups", {})

    print(f"{manifest['record_title']} ({manifest['version']})")
    print(f"record id: {manifest.get('zenodo_record_id') or 'NOT PUBLISHED YET'}\n")

    print("groups")
    for name, keys in groups.items():
        total = sum(assets[k].get("bytes") or 0 for k in keys if k in assets)
        size_text = f"{total / 2**20:.0f} MiB" if total else "size unknown"
        print(f"  --{name:<22} {size_text:>12}  {', '.join(keys)}")

    print("\nassets")
    for key, asset in assets.items():
        size = asset.get("bytes")
        size_text = f"{size / 2**20:.1f} MiB" if size else "size unknown"
        ready = "ready" if asset.get("sha256") else "NOT BUILT"
        print(f"  {key:<26} {size_text:>12}  [{ready}]  -> {asset['destination']}")
        print(f"  {'':<26} {asset.get('description', '')}")

    print(f"\ndefault: {', '.join(manifest.get('default_assets', []))}")


def resolve_selection(
    args: argparse.Namespace, manifest: Dict[str, object]
) -> List[str]:
    assets: Dict[str, Dict[str, object]] = manifest["assets"]
    groups: Dict[str, List[str]] = manifest.get("groups", {})

    requested: List[str] = []
    from_group = set()

    if args.all:
        requested.extend(assets)
        from_group.update(assets)

    for name, keys in groups.items():
        if getattr(args, name.replace("-", "_"), False):
            requested.extend(keys)
            from_group.update(keys)

    explicit: List[str] = []
    if args.assets:
        explicit = [k.strip() for k in args.assets.split(",") if k.strip()]
        requested.extend(explicit)

    if not requested:
        requested = list(manifest.get("default_assets", assets))

    unknown = [k for k in requested if k not in assets]
    if unknown:
        sys.exit(f"Unknown asset(s): {', '.join(unknown)}. Known: {', '.join(assets)}")

    # Keep manifest order, drop duplicates from overlapping groups.
    selected = [k for k in assets if k in set(requested)]

    # An archive that has not been built yet has no checksum. Skip it quietly when it
    # arrived via a group or --all; fail loudly when it was named outright.
    unbuilt = [k for k in selected if not assets[k].get("sha256")]
    named_unbuilt = [k for k in unbuilt if k in explicit]
    if named_unbuilt:
        sys.exit(
            f"Not published yet: {', '.join(named_unbuilt)}. "
            "Build the archive and record its checksum in the manifest first."
        )
    if unbuilt:
        print(f"Skipping (not published yet): {', '.join(unbuilt)}")
        selected = [k for k in selected if k not in set(unbuilt)]

    if not selected:
        sys.exit("Nothing to download.")
    return selected


def main() -> None:
    manifest_path, manifest = load_manifest(sys.argv[1:])
    parser = build_parser(manifest.get("groups", {}))
    args = parser.parse_args()

    if args.manifest.resolve() != manifest_path:
        manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))

    if args.list:
        describe(manifest)
        return

    record_id = manifest.get("zenodo_record_id")
    if not record_id:
        sys.exit(
            "zenodo_record_id is not set in the manifest.\n"
            "Publish the Zenodo record first, then record its numeric id."
        )

    selected = resolve_selection(args, manifest)
    for key in selected:
        download_asset(
            key,
            manifest["assets"][key],
            record_id,
            args.base_url,
            args.project_root.resolve(),
            args.force,
        )

    print(f"\nDone. Downloaded and verified: {', '.join(selected)}")


if __name__ == "__main__":
    main()
