#!/usr/bin/env python3
"""
test_preoptimisation.py

1) Generates a VAR(9,2) model (connectivity “tnet9x”) via simulate_model().
2) Runs the pre-optimisation procedure.
3) Saves diagnostic plots in py_ssdi/results/preoptimisation_results/.
"""

import sys, os, pickle
import matplotlib.pyplot as plt

# ── package path ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from py_ssdi.simulate_model import simulate_model
from py_ssdi.preoptimisation.preoptimise import run_preoptimisation
from py_ssdi.visualisation.plot_utils import (
    plot_model_matrices,
    plot_optimisation_runs,
    plot_distance_matrix,
)

# -----------------------------------------------------------
def main() -> None:
    # 1) generate VAR(9,2)
    print("=== Generating VAR(9,2) model (tnet9x) ===")
    model_path = simulate_model(
        "VAR",
        n=9,
        r_var=2,
        connectivity_name="tnet9x",
        seed=42,
        spectral_norm=0.90,
        decay=1.0,
    )

    # 2) run pre-optimisation
    print("\n=== Running pre-optimisation ===")
    preopt_path = run_preoptimisation(
        model_path,
        nrunsp=100,
        niters_p=10_000,
        init_step=0.5,
        tol=[1e-8, 1e-8, 1e-9],
        ctol=1e-3,
        compute_history=True,
        parallel=False,          # parallel not yet implemented
    )
    print("\nPre-optimisation results saved to:", preopt_path)

    # 3) diagnostic plots
    results_dir = os.path.join(
        PROJECT_ROOT, "py_ssdi", "results", "preoptimisation_results"
    )
    os.makedirs(results_dir, exist_ok=True)

    # --- 3a plot model matrices ------------------------------------------
    with open(model_path, "rb") as f:
        md = pickle.load(f)

    connectivity = md["connectivity"]          # (n,n,r)
    A_var_cor    = md["A_var_cor"]             # (n,n,r)
    C_ss         = md["C_ss"]                  # (n, n*r)
    A_ss         = md["A_ss"]                  # (n*r, n*r)
    V_id         = md["V_identity"]            # identity (n,n)
    K_ss         = md["K_ss"]                  # (n*r, n)

    plot_model_matrices(connectivity, A_var_cor, C_ss, A_ss, V_id, K_ss)
    for idx, fig_num in enumerate(plt.get_fignums(), 1):
        plt.figure(fig_num)
        plt.savefig(os.path.join(results_dir, f"model_matrix_{idx}.png"))
    plt.close("all")

    # --- 3b plot optimisation histories & distance matrices ---------------
    with open(preopt_path, "rb") as f:
        preopt_data = pickle.load(f)

    for m_key, m_res in preopt_data["results"].items():
        histp = m_res["histp"]
        goptp = m_res["goptp"]

        # distance matrix
        plt.figure()
        plot_distance_matrix(goptp)
        plt.title(f"goptp at {m_key}")
        plt.savefig(os.path.join(results_dir, f"preopt_goptp_{m_key}.png"))
        plt.close()

        # DD histories
        plt.figure()
        plot_optimisation_runs(histp)
        plt.title(f"Pre-opt DD histories ({m_key})")
        plt.savefig(os.path.join(results_dir, f"preopt_histories_{m_key}.png"))
        plt.close()

    print("\nPlots saved in:", results_dir)


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
