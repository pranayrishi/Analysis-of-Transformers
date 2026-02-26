"""
Attention-matrix visualisation helpers (Figures 5e-g, 7b-c, 8c-d).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_attention_heatmap(A, ax=None, title="", cmap="viridis", vmin=0, vmax=1):
    """Plot a single (seq_len × seq_len) attention matrix as a heatmap.

    Parameters
    ----------
    A : ndarray (seq_len, seq_len)
    """
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax


def plot_attention_seeds(
    attention_last_rows,
    time_labels=None,
    ax=None,
    title="",
    cmap="viridis",
):
    """Plot last-row attention weights across seeds as a 2-D heatmap.

    Parameters
    ----------
    attention_last_rows : ndarray (n_seeds, seq_len)
        Last-query attention vector for each seed.
    time_labels : list[str] | None
        Column labels (token positions).
    """
    if ax is None:
        _, ax = plt.subplots()

    n_seeds, seq_len = attention_last_rows.shape
    im = ax.imshow(attention_last_rows, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Seed")
    ax.set_yticks(range(n_seeds))
    ax.set_xticks(range(seq_len))
    if time_labels is not None:
        ax.set_xticklabels(time_labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return ax
