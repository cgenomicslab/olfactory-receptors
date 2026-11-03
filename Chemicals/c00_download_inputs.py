"""Download COCONUT (08-2026) and the NCBI taxonomy of 2026-08-01 into data/, and check both.

    python c00_download_inputs.py                  # both
    python c00_download_inputs.py --only coconut   # or --only taxonomy

Every download is checked against the sha256 recorded below and the run stops on a mismatch.
The taxonomy is the dated NCBI archive taxdmp_2026-08-01.zip, the monthly dump that matches
the COCONUT 08-2026 release. It is converted once into data/taxa.sqlite, the ete4 database
that c07, c08 and c10 open; none of them downloads a taxonomy of its own.

Files already present are checked and kept. Archives are deleted after extraction unless
--keep-archives is given.
"""
import argparse
import hashlib
import os
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import chemspace as cs

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

COCONUT_URL = "https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip"
COCONUT_ZIP = DATA / "coconut_csv-08-2026.zip"
COCONUT_CSV = DATA / "coconut_csv-08-2026.csv"
COCONUT_ZIP_SHA256 = "4a03e9224527d5067a30af8ca98c9bcb8a199f065f66aef394b08365b7a5e8fe"
COCONUT_CSV_SHA256 = "29959ca779c5f4eda6dea14a0ee9518c8d5ba8f93acde1bcded316634e562d9b"

TAXDUMP_URL = ("https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump_archive/"
               + cs.TAXDUMP_ARCHIVE)
TAXDUMP_ZIP = DATA / cs.TAXDUMP_ARCHIVE
TAXDUMP_SHA256 = "2e44982c8040182d6fbe25888bcd23d2355dfd3148d2abd54e5a6b380312b02e"

# The three files ete4 reads to build its database. merged.dmp maps retired taxids to
# their replacements.
TAXDUMP_MEMBERS = ["nodes.dmp", "names.dmp", "merged.dmp"]


def sha256(path):
    """Return the sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify(path, expected):
    """Stop the run if a file's sha256 is not the recorded one."""
    found = sha256(path)
    if found != expected:
        raise SystemExit(
            f"checksum mismatch for {path.name}\n"
            f"  expected {expected}\n"
            f"  found    {found}\n"
            "  delete the file and run this script again"
        )
    print(f"  sha256 ok  {path.name}")


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
    """Download, check and unzip the COCONUT CSV."""
    if COCONUT_CSV.exists():
        print(f"coconut: already have {COCONUT_CSV.name}")
        verify(COCONUT_CSV, COCONUT_CSV_SHA256)
        return
    if not COCONUT_ZIP.exists():
        fetch(COCONUT_URL, COCONUT_ZIP)
    verify(COCONUT_ZIP, COCONUT_ZIP_SHA256)
    with zipfile.ZipFile(COCONUT_ZIP) as archive:
        archive.extract(COCONUT_CSV.name, DATA)
    verify(COCONUT_CSV, COCONUT_CSV_SHA256)
    if not keep_archive:
        COCONUT_ZIP.unlink()
    print(f"coconut: {COCONUT_CSV.name} ready")


def build_taxa_database(archive):
    """
    Convert the NCBI dump into ete4's sqlite database at cs.TAXA_DB.

    ete4 reads the dump from a tar file and writes scratch tables to the working
    directory, so both happen inside a temporary directory.
    """
    with tempfile.TemporaryDirectory(dir=DATA) as scratch:
        scratch = Path(scratch)
        dump_tar = scratch / "taxdump.tar"
        with zipfile.ZipFile(archive) as source, tarfile.open(dump_tar, "w") as target:
            for member in TAXDUMP_MEMBERS:
                source.extract(member, scratch)
                target.add(scratch / member, arcname=member)
                (scratch / member).unlink()

        from ete4 import NCBITaxa

        working_directory = os.getcwd()
        os.chdir(scratch)
        try:
            NCBITaxa(dbfile=str(cs.TAXA_DB), taxdump_file=str(dump_tar), update=False)
        finally:
            os.chdir(working_directory)

    cs.TAXA_DB_SOURCE.write_text(f"{cs.TAXDUMP_ARCHIVE}\n{TAXDUMP_SHA256}\n")


def get_taxonomy(keep_archive):
    """Download and check the dated NCBI dump, and build data/taxa.sqlite from it."""
    if cs.taxonomy_is_built():
        print(f"taxonomy: already have {cs.TAXA_DB.name} built from {cs.TAXDUMP_ARCHIVE}")
        return
    if not TAXDUMP_ZIP.exists():
        fetch(TAXDUMP_URL, TAXDUMP_ZIP)
    verify(TAXDUMP_ZIP, TAXDUMP_SHA256)
    build_taxa_database(TAXDUMP_ZIP)
    if not keep_archive:
        TAXDUMP_ZIP.unlink()
    print(f"taxonomy: {cs.TAXA_DB.name} ready, from {cs.TAXDUMP_ARCHIVE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=["coconut", "taxonomy"],
                        help="fetch just one of the two")
    parser.add_argument("--keep-archives", action="store_true",
                        help="keep the two .zip files after extracting (about 310 MB)")
    args = parser.parse_args()

    DATA.mkdir(exist_ok=True)
    if args.only != "taxonomy":
        get_coconut(args.keep_archives)
    if args.only != "coconut":
        get_taxonomy(args.keep_archives)


if __name__ == "__main__":
    main()
