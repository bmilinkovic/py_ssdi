#!/usr/bin/env python3
"""
plot_optimisation_results.py

Load saved optimisation results and create composite visualizations.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from py_ssdi.visualisation.plot_utils import (
    plot_composite_histories,
    plot_composite_goptp
)

def main():
    # Load optimisation results
    opt_path = "py_ssdi/results/optimisation_results/opt_dd_all_VAR_n9_r2_tnet9x.pkl"
    print(f"\nLoading optimisation results from: {opt_path}")
    with open(opt_path, "rb") as f:
        opt_results = pickle.load(f)
    print("Optimisation results loaded successfully")
    print(f"Available keys in opt_results: {list(opt_results.keys())}")

    # Extract from nested 'results' key
    results = opt_results['results']
    print(f"Available keys in opt_results['results']: {list(results.keys())}")

    # Create output directory if it doesn't exist
    output_dir = "py_ssdi/results/optimisation_results/figures"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving figures to: {output_dir}")

    # Remap 'gopto' to 'goptp' and 'ohisto' to 'histp' for compatibility with the plotting functions
    results_for_plot = {}
    for m_key, m_res in results.items():
        m_res_copy = m_res.copy()
        if 'gopto' in m_res_copy:
            m_res_copy['goptp'] = m_res_copy['gopto']
        if 'ohisto' in m_res_copy:
            m_res_copy['histp'] = m_res_copy['ohisto']
        results_for_plot[m_key] = m_res_copy

    # Create and save composite histories plot
    fig_hist = plot_composite_histories(results_for_plot)
    fig_hist.savefig(os.path.join(output_dir, 'opt_histories_all.png'), 
                     dpi=300, bbox_inches='tight')
    plt.close(fig_hist)
    print(f"Saved composite histories plot to: opt_histories_all.png")

    # Create and save composite goptp plot using remapped results
    fig_goptp = plot_composite_goptp(results_for_plot)
    fig_goptp.savefig(os.path.join(output_dir, 'opt_goptp_all.png'), 
                      dpi=300, bbox_inches='tight')
    plt.close(fig_goptp)
    print(f"Saved composite goptp plot to: opt_goptp_all.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
