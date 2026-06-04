"""
Download Rossmann Store Sales data from Kaggle if not already present.

Usage: python download_data.py

Requires Kaggle credentials at ~/.kaggle/access_token (new format) or
~/.kaggle/kaggle.json (legacy format). See data/README.md for details.
"""

import os
import zipfile
from pathlib import Path

COMPETITION = "rossmann-store-sales"
RAW_DIR = Path("data/raw")
EXPECTED_FILES = ["train.csv", "test.csv", "store.csv"]


def all_files_present():
    """Return True if all expected raw data files already exist."""
    return all((RAW_DIR / f).exists() for f in EXPECTED_FILES)


def download():
    """Download and unzip competition files from Kaggle."""
    from kaggle.api.kaggle_api_extended import KaggleApiExtended

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApiExtended()
    api.authenticate()

    print(f"Downloading {COMPETITION} data to {RAW_DIR}/...")
    api.competition_download_files(COMPETITION, path=RAW_DIR, quiet=False)

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()
        print("Extraction complete, zip removed.")


def main():
    if all_files_present():
        print("Data already present, nothing to download.")
        return

    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
    print(f"Missing files: {missing}")
    download()

    if all_files_present():
        print("All files downloaded successfully:")
        for f in EXPECTED_FILES:
            size_mb = (RAW_DIR / f).stat().st_size / 1_000_000
            print(f"  {f}: {size_mb:.1f} MB")
    else:
        # TO REVIEW: download appeared to succeed but expected files still missing
        # — check if Kaggle changed the zip contents or file names
        still_missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
        print(f"WARNING: still missing after download: {still_missing}")
        print(f"Files present in {RAW_DIR}/:")
        for p in sorted(RAW_DIR.iterdir()):
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
