#!/usr/bin/env python3
"""
test_data.py

Inspect the pyspi_data.pkl database, print available dataset names, and filter for EEG-related datasets.
"""

import pickle
import os

import numpy as np

DATA_PATH = "py_ssdi/data/pyspi_data.pkl"


def main():
    print(f"Loading database from: {DATA_PATH}")
    with open(DATA_PATH, "rb") as f:
        db = pickle.load(f)
    print(f"Loaded {len(db)} datasets.")

    # Print all dataset names
    print("\nAll dataset names:")
    for name in list(db.keys())[:20]:  # print first 20 for brevity
        print("  ", name)
    if len(db) > 20:
        print(f"  ... (total {len(db)} datasets)")

    # Find EEG-related datasets
    eeg_datasets = [k for k in db if 'eeg' in k.lower() or any('eeg' in str(lbl).lower() for lbl in db[k].get('labels', []))]
    print(f"\nFound {len(eeg_datasets)} EEG-related datasets:")
    for name in eeg_datasets:
        print("  ", name)

    # Show info about the first EEG dataset
    if eeg_datasets:
        eeg_name = eeg_datasets[0]
        eeg_data = db[eeg_name]['data']
        eeg_labels = db[eeg_name].get('labels', [])
        print(f"\nFirst EEG dataset: {eeg_name}")
        print(f"  Data shape: {eeg_data.shape} (processes x observations)")
        print(f"  Labels: {eeg_labels}")
        print(f"  Data preview (first 5 rows, 5 cols):\n{eeg_data[:5, :5]}")
    else:
        print("No EEG datasets found.")

if __name__ == "__main__":
    main() 