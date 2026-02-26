"""
Navier-Stokes flow past a cylinder — data loading (Section 4.3).

The paper uses the dataset from Geneva & Zabaras (2022) [21]:
  https://github.com/zabaras/transformer-physx
  Zenodo: https://zenodo.org/records/5148524

Observation: u_x at spatial location (x=35, y=45).

Supports three formats:
  1. HDF5 from Geneva & Zabaras (original)  — ``cylinder_{split}.h5``
  2. Converted ``.npz`` files                — ``cylinder_ReXXX.npz``
  3. Stuart-Landau surrogate (fallback)

If the dataset is unavailable, a Stuart-Landau surrogate produces
qualitatively similar vortex-shedding signals parameterised by Re.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp


# ═══════════════════════════════════════════════════════════════════
#  HDF5 loader (original Geneva & Zabaras format)
# ═══════════════════════════════════════════════════════════════════

def load_navier_stokes_h5(h5_path: str, obs_x: int = 35, obs_y: int = 45):
    """Load trajectories from a single Geneva & Zabaras HDF5 file.

    The HDF5 structure:
        <Re_key>/ux   (T, 64, 128)
        <Re_key>/uy   (T, 64, 128)
        <Re_key>/p    (T, 64, 128)

    Where Re = float(key).

    Returns
    -------
    trajectories : dict[float, ndarray (T, 1)]
    full_fields  : dict[float, ndarray (T, Ny, Nx, 3)]
    """
    import h5py

    trajectories, full_fields = {}, {}

    with h5py.File(h5_path, "r") as f:
        for key in sorted(f.keys(), key=lambda k: float(k)):
            Re = float(key)
            grp = f[key]
            ux = np.array(grp["ux"])  # (T, 64, 128)
            trajectories[Re] = ux[:, obs_y, obs_x][:, None]

            if "uy" in grp and "p" in grp:
                uy = np.array(grp["uy"])
                p = np.array(grp["p"])
                full_fields[Re] = np.stack([ux, uy, p], axis=-1)

    return trajectories, full_fields


def convert_h5_to_npz(h5_path: str, output_dir: str):
    """Convert Geneva & Zabaras HDF5 to individual .npz files."""
    import h5py

    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            Re = float(key)
            grp = f[key]
            save_dict = {"Re": Re, "ux": np.array(grp["ux"])}
            if "uy" in grp:
                save_dict["uy"] = np.array(grp["uy"])
            if "p" in grp:
                save_dict["p"] = np.array(grp["p"])

            fname = f"cylinder_Re{int(Re)}.npz"
            np.savez(os.path.join(output_dir, fname), **save_dict)
            print(f"  Saved {fname} (T={save_dict['ux'].shape[0]})")


# ═══════════════════════════════════════════════════════════════════
#  Unified loader: tries HDF5 first, then .npz
# ═══════════════════════════════════════════════════════════════════

def load_navier_stokes_data(data_dir: str, obs_x: int = 35, obs_y: int = 45):
    """Load pre-generated Navier-Stokes trajectories.

    Tries:
      1. HDF5 files (``cylinder_*.h5``) — original Geneva & Zabaras format.
      2. NPZ files (``cylinder_Re*.npz``) — converted format.

    Returns
    -------
    trajectories : dict[float, ndarray (T, 1)]
    full_fields  : dict[float, ndarray (T, Ny, Nx, 3)]
    """
    trajectories, full_fields = {}, {}

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            "Download from https://zenodo.org/records/5148524"
        )

    # Try HDF5 files first (.h5 or .hdf5)
    h5_files = [f for f in os.listdir(data_dir)
                if f.endswith(".h5") or f.endswith(".hdf5")]
    if h5_files:
        print(f"  Loading HDF5 files: {h5_files}")
        for h5f in sorted(h5_files):
            t, ff = load_navier_stokes_h5(
                os.path.join(data_dir, h5f), obs_x, obs_y
            )
            trajectories.update(t)
            full_fields.update(ff)
        if trajectories:
            print(f"  Loaded {len(trajectories)} Re values from HDF5")
            return trajectories, full_fields

    # Fall back to .npz files
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        data = np.load(os.path.join(data_dir, fname))
        Re = float(data["Re"]) if "Re" in data else _parse_re(fname)
        ux = data["ux"]  # (T, Ny, Nx)
        trajectories[Re] = ux[:, obs_y, obs_x][:, None]
        if {"ux", "uy", "p"}.issubset(data.files):
            full_fields[Re] = np.stack(
                [data["ux"], data["uy"], data["p"]], axis=-1
            )

    if not trajectories:
        raise FileNotFoundError(
            f"No .h5 or .npz data files found in {data_dir}"
        )

    return trajectories, full_fields


def _parse_re(filename: str) -> float:
    import re as _re
    m = _re.search(r"[Rr][Ee](\d+\.?\d*)", filename)
    return float(m.group(1)) if m else 0.0


# ═══════════════════════════════════════════════════════════════════
#  Stuart-Landau surrogate (used when CFD dataset is unavailable)
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_cylinder_data(
    Re_values=None, t_end: float = 200.0, dt: float = 1.0,
    transient: float = 50.0, seed: int = 42,
):
    """Stuart-Landau oscillator parameterised by Reynolds number.

    dz/dt = (sigma + i omega) z - (1 + i beta) |z|^2 z

    sigma and omega depend on Re via Strouhal-number scaling:
      St ~ 0.198 (1 - 19.7/Re)   for 250 < Re < 2e5.

    NOTE: this is a SURROGATE. Mark results accordingly.

    Returns
    -------
    trajectories : dict[float, ndarray (T, 1)]
    """
    if Re_values is None:
        Re_values = [100, 200, 300, 400, 500, 600, 750]

    rng = np.random.default_rng(seed)
    trajectories = {}

    for Re in Re_values:
        St = 0.198 * (1 - 19.7 / Re) if Re > 50 else 0.1
        omega = 2 * np.pi * St
        sigma = 0.1 * (1 - 47.0 / Re) if Re > 47 else -0.1
        beta = 0.5

        def stuart_landau(t, y, _s=sigma, _w=omega, _b=beta):
            z = y[0] + 1j * y[1]
            r2 = abs(z) ** 2
            dzdt = (_s + 1j * _w) * z - (1 + 1j * _b) * r2 * z
            return [dzdt.real, dzdt.imag]

        z0 = [0.1 * rng.standard_normal(), 0.1 * rng.standard_normal()]
        t_eval = np.arange(0, t_end, dt)
        sol = solve_ivp(stuart_landau, [0, t_end], z0,
                        t_eval=t_eval, method="RK45",
                        rtol=1e-8, atol=1e-10)

        if sol.success:
            mask = sol.t >= transient
            base_flow = 1.0 - 0.5 * np.exp(-Re / 200.0)
            signal = sol.y[0, mask] + base_flow
            trajectories[float(Re)] = signal[:, None]

    return trajectories
