#!/usr/bin/env python3
"""
Script to unpack dataset zips from data/raw/datasets/ into data/raw/unpacked/.
Run this when you need to access the zipped datasets.
"""

import os
import zipfile
import shutil

def unpack_datasets():
    raw_dir = "data/raw"
    datasets_dir = os.path.join(raw_dir, "datasets")
    unpacked_dir = os.path.join(raw_dir, "unpacked")

    if not os.path.exists(datasets_dir):
        print(f"Error: {datasets_dir} does not exist.")
        return

    os.makedirs(unpacked_dir, exist_ok=True)

    for file in os.listdir(datasets_dir):
        if file.endswith(".zip"):
            zip_path = os.path.join(datasets_dir, file)
            extract_to = os.path.join(unpacked_dir, file[:-4])  # Remove .zip extension
            print(f"Unpacking {zip_path} to {extract_to}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Unpacked {file}.")

    print("All zips unpacked. Check data/raw/unpacked/ for contents.")

if __name__ == "__main__":
    unpack_datasets()