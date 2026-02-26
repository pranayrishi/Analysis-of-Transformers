"""Re-run ONLY the latent analysis (Figures 7a, 7b-c, 8) for Chafee-Infante.

Skips Figures 6a, 6b, 6c (already correct).
Uses dense data (500 trajs × 80 time steps) for manifold visualisation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp_chafee_infante import (
    prepare_data, plot_latent_analysis,
)

if __name__ == "__main__":
    print("Preparing Chafee-Infante data...")
    trajs_modes, train, val, test, t_eval = prepare_data()

    print("\n=== Latent analysis (Figures 7-8) ===")
    plot_latent_analysis(train, val, test, trajs_modes)

    print("\nDone — Figures 7a, 7b-c, 8 regenerated.")
