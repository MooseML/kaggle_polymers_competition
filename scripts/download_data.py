#!/usr/bin/env python3
"""
Kaggle downloader for the NeurIPS Open Polymer Prediction 2025 competition.

Usage:
    python scripts/download_data.py

What it does:
  1. Checks for Kaggle CLI and your API key (kaggle.json).
  2. Downloads the competition files into ./data
  3. Unzips all downloaded .zip files into ./data
"""

from __future__ import annotations
import os
import sys
import shutil
import subprocess
from pathlib import Path
import zipfile

# --- USER CONFIG (edit only if needed) ---------------------------------------
COMPETITION_SLUG = "neurips-open-polymer-prediction-2025"
DEST_DIR = Path("data")              # where files will be downloaded/extracted
UNZIP = True                         # auto-unzip all .zip files into DEST_DIR

# If your kaggle.json is in a non-standard folder, set this to that folder path.
# Example (Windows): r"C:\Users\you\.kaggle"
# Example (macOS/Linux): "/Users/you/.kaggle"
KAGGLE_CONFIG_DIR_OVERRIDE: str | None = None
# -----------------------------------------------------------------------------

def ensure_kaggle_cli():
    if shutil.which("kaggle") is None:
        sys.exit(
            "ERROR: Kaggle CLI not found.\n"
            "Install it with:\n"
            "    pip install kaggle\n"
            "Then set up your API key (see instructions below)."
        )


def ensure_kaggle_credentials():
    # Respect override if provided
    if KAGGLE_CONFIG_DIR_OVERRIDE:
        os.environ["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR_OVERRIDE

    possible_paths = []
    if "KAGGLE_CONFIG_DIR" in os.environ:
        possible_paths.append(Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json")
    home = Path.home()
    possible_paths += [
        home / ".kaggle" / "kaggle.json",  # macOS/Linux default
        Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json",  # Windows
    ]

    if not any(p.exists() for p in possible_paths):
        msg = (
            "ERROR: Kaggle API key (kaggle.json) not found.\n\n"
            "Option A) Set it up in the default location:\n"
            "  - Linux/macOS: ~/.kaggle/kaggle.json\n"
            "  - Windows: %USERPROFILE%\\.kaggle\\kaggle.json\n\n"
            "Option B) Put it anywhere and set KAGGLE_CONFIG_DIR to that folder, e.g.:\n"
            "  - Linux/macOS: export KAGGLE_CONFIG_DIR=/path/to/folder\n"
            "  - Windows (PowerShell): $env:KAGGLE_CONFIG_DIR='C:\\path\\to\\folder'\n\n"
            "Option C) Edit this script: set KAGGLE_CONFIG_DIR_OVERRIDE to your folder path.\n\n"
            "Get your key from https://www.kaggle.com/settings/account (Create New API Token)."
        )
        sys.exit(msg)

    # Let the Kaggle CLI validate creds format/permissions
    try:
        subprocess.run(
            ["kaggle", "config", "view"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        sys.exit(
            "ERROR: Kaggle credentials found but invalid or unreadable.\n"
            "Re-download your API token and ensure permissions are correct.\n"
            "Linux/macOS: chmod 600 ~/.kaggle/kaggle.json"
        )


def download_competition(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", slug, "-p", str(dest)]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"ERROR: Kaggle download failed (exit code {e.returncode}).\n"
            "- Make sure you have accepted the competition rules on the Kaggle website.\n"
            "- Check your internet connection and Kaggle account access.\n"
            "- Try running the command manually to see detailed errors."
        )


def unzip_all_in(dest: Path) -> None:
    zips = list(dest.glob("*.zip"))
    if not zips:
        print("No .zip files found to extract (maybe already unzipped).")
        return
    for z in zips:
        print(f"Extracting: {z.name}")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dest)
    print(f"Extracted to: {dest.resolve()}")


def main():
    ensure_kaggle_cli()
    ensure_kaggle_credentials()
    download_competition(COMPETITION_SLUG, DEST_DIR)
    if UNZIP:
        unzip_all_in(DEST_DIR)
    print("\nDone. Check the 'data/' folder for files.\n"
          "If filenames differ from your training script expectations, "
          "rename them or update your config.")


if __name__ == "__main__":
    main()
