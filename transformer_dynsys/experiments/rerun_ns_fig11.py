"""Re-run ONLY Figure 11 (NS MSE comparison) with 10 seeds.

Skips Figures 9 and 10 (already correct).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp_navier_stokes import prepare_data, plot_figure11

if __name__ == "__main__":
    print("Preparing Navier-Stokes data...")
    trajs, Re_values = prepare_data(use_re_input=False)
    print(f"  Loaded {len(Re_values)} Re values: {Re_values[:3]}...{Re_values[-3:]}")

    print("\n=== MSE comparison (Figure 11) — 10 seeds ===")
    plot_figure11(trajs, Re_values)

    print("\nDone — Figure 11 regenerated with 10 seeds.")
