#!/usr/bin/env python3
"""
Master runner for reproducing all experiments in
"A Mechanistic Analysis of Transformers for Dynamical Systems"
(Duthé et al., arXiv:2512.21113v1, December 2025).

Usage
-----
  python run_all.py            # run everything (takes hours)
  python run_all.py --phase 2  # run Phase 2 only (SDOF + 2DOF)
  python run_all.py --phase 3  # run Phase 3 only (Van der Pol)
  python run_all.py --phase 4  # run Phase 4 only (Chafee-Infante)
  python run_all.py --phase 5  # run Phase 5 only (Navier-Stokes)
"""

import argparse
import time


def run_phase2():
    print("\n" + "=" * 70)
    print("PHASE 2: LINEAR DYNAMICAL SYSTEMS (Section 3)")
    print("=" * 70)
    from experiments.exp_sdof import main as sdof_main
    from experiments.exp_2dof import main as twodof_main
    sdof_main()
    twodof_main()


def run_phase3():
    print("\n" + "=" * 70)
    print("PHASE 3: VAN DER POL OSCILLATOR (Section 4.1)")
    print("=" * 70)
    from experiments.exp_vanderpol import main as vdp_main
    vdp_main()


def run_phase4():
    print("\n" + "=" * 70)
    print("PHASE 4: CHAFEE-INFANTE (Section 4.2)")
    print("=" * 70)
    from experiments.exp_chafee_infante import main as ci_main
    ci_main()


def run_phase5():
    print("\n" + "=" * 70)
    print("PHASE 5: NAVIER-STOKES (Section 4.3)")
    print("=" * 70)
    from experiments.exp_navier_stokes import main as ns_main
    ns_main()


PHASES = {2: run_phase2, 3: run_phase3, 4: run_phase4, 5: run_phase5}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", type=int, default=None,
                        choices=[2, 3, 4, 5],
                        help="Run a single phase (default: all)")
    args = parser.parse_args()

    t0 = time.time()

    if args.phase:
        PHASES[args.phase]()
    else:
        for phase_fn in PHASES.values():
            phase_fn()

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
