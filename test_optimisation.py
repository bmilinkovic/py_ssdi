#!/usr/bin/env python3
"""
test_optimisation.py

1) Load the saved VAR(9,2) model and its pre-optimisation results.
2) Run final SSDI optimisation for m = 2 … 7.
3) Save consolidated results in py_ssdi/results/optimisation_results/.
"""
import os, sys, time, pickle
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# run_optimisation.py lives in py_ssdi/optimisation/
from py_ssdi.optimisation.optimise import run_optimisation


def abspath(*parts) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def main() -> None:
    # ── paths ───────────────────────────────────────────────
    model_path = abspath("py_ssdi", "results", "model_parameters",
                         "VAR_n9_r2_tnet9x.pkl")
    preopt_path = abspath("py_ssdi", "preoptimisation", "results", "preoptimisation_results",
                          "preopt_VAR_n9_r2_tnet9x.pkl")

    print("Model path :", model_path)
    print("Pre-opt path:", preopt_path)

    # ── fetch fres from the model pickle ────────────────────
    with open(model_path, "rb") as f:
        md = pickle.load(f)

    print("‖V_identity – I‖₂ =", np.linalg.norm(md["V_identity"] - np.eye(md["C_ss"].shape[0])))
    fres_len = md["fres"]          # power-of-two length chosen by var2fres
    ierr     = md["ierr"]
    print(f"Using frequency resolution fres = {fres_len}  (integral error {ierr:.2e})")

    omega = np.linspace(0.0, np.pi, fres_len + 1)   # 0 … π inclusive

    # ── loop over macroscopic dimensions ────────────────────
    all_results: dict[str, dict] = {"results": {}}

    gdtol = (1e-12, 1e-10, 1e-6)  # stol, dtol, gtol
    for mdim in range(2, 8):       # m = 2 … 7
        print(f"\n=== Optimising for m = {mdim} ===")
        res = run_optimisation(
            model_path=model_path,
            preopt_path=preopt_path,
            mdim=mdim,
            fres=omega,            # ← pass the array, not the integer
            niters_o=10_000,
            init_step=0.1,
            tol=gdtol,
            ctol=1e-6,
            compute_history=True,
            parallel=False,        # stub not yet implemented
        )
        all_results["results"][f"m={mdim}"] = res

    # ── save everything ─────────────────────────────────────
    results_dir = abspath("py_ssdi", "results", "optimisation_results")
    os.makedirs(results_dir, exist_ok=True)

    base     = os.path.splitext(os.path.basename(model_path))[0]
    out_path = os.path.join(results_dir, f"opt_dd_all_{base}.pkl")

    all_results.update(
        {
            "model_path":  model_path,
            "preopt_path": preopt_path,
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
            "fres":        fres_len,
            "fres_ierr":   ierr,
        }
    )

    with open(out_path, "wb") as f:
        pickle.dump(all_results, f)

    print("\nAll optimisation results saved to:", out_path)
    print("Diagnostic plotting hooks coming in the next phase.")


if __name__ == "__main__":
    main()

