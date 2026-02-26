"""
Standalone MLP baseline (time-one map: current state → next state).
Used as the comparison model in Section 4.
"""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """Two-layer feedforward network.

    Parameters
    ----------
    d_input : int
        Current-state dimension.
    d_output : int
        Next-state dimension.
    hidden_size : int
        Width of the single hidden layer.
    activation : str
        ``'relu'`` or ``'tanh'``.
    """

    def __init__(
        self,
        d_input: int,
        d_output: int,
        hidden_size: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        act = nn.ReLU() if activation == "relu" else nn.Tanh()
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden_size),
            act,
            nn.Linear(hidden_size, d_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, d_input) → (B, d_output)"""
        return self.net(x)
