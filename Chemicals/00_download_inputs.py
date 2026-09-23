"""Download the two inputs that are too large to commit.

COCONUT is needed by 01, 07, 08, 09, 10 and 12. The NCBI taxonomy is needed by 12 only.
Both are skipped if their extracted files are already present.
"""
import argparse
import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

COCONUT_URL = "https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"

TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
TAXDUMP_MEMBERS = ["nodes.dmp", "names.dmp"]


def fetch(url, destination):
    """Stream a download to disk, reporting progress."""
    print(f"  {url}", flush=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        seen = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            seen += len(chunk)
            if total:
                print(f"\r  {100 * seen / total:5.1f}%  {seen / 1e6:.0f} MB", end="", flush=True)
    print(flush=True)


def get_coconut(keep_archive):
    if COCONUT_CSV.exists():
        print(f"coconut: already have {COCONUT_CSV.name}")
        return
    archive = DATA / Path(COCONUT_URL).name
    if not archive.exists():
        fetch(COCONUT_URL, archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA)
    if not keep_archive:
        archive.unlink()
    print(f"coconut: {COCONUT_CSV.name} ready")


def get_taxdump(keep_archive):
    if all((DATA / m).exists() for m in TAXDUMP_MEMBERS):
        print("taxonomy: already have nodes.dmp and names.dmp")
        return
    archive = DATA / Path(TAXDUMP_URL).name
    if not archive.exists():
        fetch(TAXDUMP_URL, archive)
    with tarfile.open(archive) as t:
        for member in TAXDUMP_MEMBERS:
            t.extract(member, DATA)
    if not keep_archive:
        archive.unlink()
    print("taxonomy: nodes.dmp and names.dmp ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["coconut", "taxonomy"],
                        help="fetch just one of the two")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the .zip and .tar.gz after extracting (311 MB)")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    if args.only != "taxonomy":
        get_coconut(args.keep_archives)
    if args.only != "coconut":
        get_taxdump(args.keep_archives)


if __name__ == "__main__":
    main()
