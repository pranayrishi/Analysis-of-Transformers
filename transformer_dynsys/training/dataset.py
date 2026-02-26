"""
PyTorch Dataset classes for overlapping sliding-window sequences.

The paper uses windows of length *n* (the delay-embedding / context length):
  input  = [u_t, u_{t+1}, …, u_{t+n-1}]   →   target = u_{t+n}
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedTimeSeriesDataset(Dataset):
    """Sliding-window dataset for transformer one-step-ahead prediction.

    Parameters
    ----------
    trajectories : ndarray | list[ndarray]
        Each array has shape ``(T_i, d_obs)``.
    window_size : int
        Number of past observations (*n*).
    target_dim : int | None
        Keep only the first *target_dim* components of the target.
    """

    def __init__(self, trajectories, window_size: int, target_dim: int | None = None):
        if not isinstance(trajectories, list):
            trajectories = [np.asarray(trajectories)]

        inputs, targets = [], []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if traj.ndim == 1:
                traj = traj[:, None]
            T = len(traj)
            for t in range(T - window_size):
                inputs.append(traj[t : t + window_size])
                tgt = traj[t + window_size]
                if target_dim is not None:
                    tgt = tgt[..., :target_dim]
                targets.append(tgt)

        self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class MLPDataset(Dataset):
    """Single-step dataset for the MLP baseline (no temporal context).

    Parameters
    ----------
    trajectories : ndarray | list[ndarray]
    input_dim, target_dim : int | None
        Optionally slice input/target dimensions.
    """

    def __init__(
        self,
        trajectories,
        input_dim: int | None = None,
        target_dim: int | None = None,
    ):
        if not isinstance(trajectories, list):
            trajectories = [np.asarray(trajectories)]

        inputs, targets = [], []
        for traj in trajectories:
            traj = np.asarray(traj, dtype=np.float64)
            if traj.ndim == 1:
                traj = traj[:, None]
            T = len(traj)
            for t in range(T - 1):
                inp = traj[t]
                tgt = traj[t + 1]
                if input_dim is not None:
                    inp = inp[..., :input_dim]
                if target_dim is not None:
                    tgt = tgt[..., :target_dim]
                inputs.append(inp)
                targets.append(tgt)

        self.inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
