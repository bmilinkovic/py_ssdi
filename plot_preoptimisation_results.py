#!/usr/bin/env python3
"""
plot_results.py

Load saved preoptimization results and create visualizations.
"""

import pickle
import os
import matplotlib.pyplot as plt
from py_ssdi.visualisation.plot_utils import (
    plot_composite_histories,
    plot_composite_goptp
)

def main():
    # Load preoptimization results
    preopt_path = "py_ssdi/preoptimisation/results/preoptimisation_results/preopt_VAR_n9_r2_tnet9x.pkl"
    print(f"\nLoading preoptimisation results from: {preopt_path}")
    with open(preopt_path, "rb") as f:
        preopt_results = pickle.load(f)
    print("Preoptimisation results loaded successfully")
    print(f"Available keys in preopt_results: {list(preopt_results.keys())}")

    # Extract from nested 'results' key
    results = preopt_results['results']
    print(f"Available keys in preopt_results['results']: {list(results.keys())}")

    # Create output directory if it doesn't exist
    output_dir = "py_ssdi/results/preoptimisation_results/figures"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving figures to: {output_dir}")

    # Create and save composite histories plot
    fig_hist = plot_composite_histories(results)
    fig_hist.savefig(os.path.join(output_dir, 'preopt_histories_all.png'), 
                     dpi=300, bbox_inches='tight')
    plt.close(fig_hist)
    print(f"Saved composite histories plot to: preopt_histories_all.png")

    # Create and save composite goptp plot
    fig_goptp = plot_composite_goptp(results)
    fig_goptp.savefig(os.path.join(output_dir, 'preopt_goptp_all.png'), 
                      dpi=300, bbox_inches='tight')
    plt.close(fig_goptp)
    print(f"Saved composite goptp plot to: preopt_goptp_all.png")

    print("\nDone.")

if __name__ == "__main__":
    main() 