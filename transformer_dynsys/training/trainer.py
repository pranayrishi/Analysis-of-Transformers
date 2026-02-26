"""
Generic training loop for both Transformer and MLP models.

Defaults:
  Adam optimiser, MSE loss, optional validation early-stopping.
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Train a model and optionally track validation loss.

    Parameters
    ----------
    model : nn.Module
    train_dataset, val_dataset : Dataset
    lr : float
    batch_size : int
    epochs : int
    device : str
    verbose : bool
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset=None,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 1000,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.verbose = verbose

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if val_dataset is not None
            else None
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_state: dict | None = None

    # ── helpers ──────────────────────────────────────────────────────
    def _is_transformer(self) -> bool:
        return hasattr(self.model, "attention")

    def _predict(self, X):
        if self._is_transformer():
            y_pred, _ = self.model(X)
        else:
            y_pred = self.model(X)
        return y_pred

    # ── main loop ───────────────────────────────────────────────────
    def train(self) -> nn.Module:
        best_val = float("inf")

        for epoch in range(self.epochs):
            # — training ──────────────────────────────────────────────
            self.model.train()
            running = 0.0
            n = 0
            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self._predict(X_b), y_b)
                loss.backward()
                self.optimizer.step()
                running += loss.item()
                n += 1
            self.train_losses.append(running / n)

            # — validation ────────────────────────────────────────────
            if self.val_loader is not None:
                val = self._evaluate(self.val_loader)
                self.val_losses.append(val)
                if val < best_val:
                    best_val = val
                    self.best_state = copy.deepcopy(self.model.state_dict())

            # — logging ───────────────────────────────────────────────
            if self.verbose and (epoch + 1) % 100 == 0:
                msg = (
                    f"Epoch {epoch+1:>5}/{self.epochs}  "
                    f"train={self.train_losses[-1]:.3e}"
                )
                if self.val_losses:
                    msg += f"  val={self.val_losses[-1]:.3e}"
                print(msg)

        # restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return self.model

    def _evaluate(self, loader) -> float:
        self.model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                loss = self.criterion(self._predict(X_b), y_b)
                running += loss.item()
                n += 1
        return running / n
