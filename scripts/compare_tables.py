#!/usr/bin/env python3
"""Compare the analysis tables in the working tree with the committed ones.

Run after rerunning the Downstream_Analysis notebooks. Numeric columns may differ by floating-point noise between
machines, so they are compared with a relative tolerance; every other column must match
exactly. Exit code 1 if any table differs.

    python scripts/compare_tables.py
"""

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TABLE_DIRS = ["Downstream_Analysis/extant_receptors/results",
              "Downstream_Analysis/ancestral_receptors/results"]
RTOL, ATOL = 1e-6, 1e-9


def committed(path):
    """The committed version of a file, or None if it is not tracked."""
    result = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True)
    return result.stdout.decode() if result.returncode == 0 else None


def differences(new, old):
    """A list of the ways two tables differ, empty if they agree."""
    if list(new.columns) != list(old.columns):
        return ["columns differ"]
    if len(new) != len(old):
        return [f"{len(old)} rows committed, {len(new)} now"]
    found = []
    for column in new.columns:
        a, b = new[column], old[column]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                found.append(f"{column}: numbers differ")
        elif not a.astype(str).equals(b.astype(str)):
            found.append(f"{column}: values differ")
    return found


def main():
    failed = 0
    for table_dir in TABLE_DIRS:
        for path in sorted((REPO / table_dir).glob("*.csv")):
            relative = path.relative_to(REPO).as_posix()
            old_text = committed(relative)
            if old_text is None:
                print(f"NEW   {relative}")
                continue
            problems = differences(pd.read_csv(path), pd.read_csv(io.StringIO(old_text)))
            if problems:
                failed += 1
                print(f"DIFF  {relative}: " + "; ".join(problems))
    print("all tables reproduce" if not failed else f"{failed} table(s) differ")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
