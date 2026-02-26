"""
Single-head causal self-attention (Section 2.2, Eqs. 4-6, Figure 1).

Implements the core attention mechanism used throughout the paper:
  Q = X̃ W_Q,   K = X̃ W_K,   V = X̃ W_V
  A = softmax(Q K^T / √d_k)          (causal mask applied)
  Z = A V
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Single-head causal (masked) self-attention.

    Parameters
    ----------
    d_input : int
        Input token dimension (d_i in the paper).
    d_k : int
        Query / Key projection dimension.
    d_v : int
        Value projection dimension.
    max_seq_len : int
        Maximum context length (for learned positional encoding).
    use_positional_encoding : bool
        Add a learned positional encoding p_i to each token.
    use_residual : bool
        Add a residual connection  Z ← Z + X̃.
    """

    def __init__(
        self,
        d_input: int,
        d_k: int,
        d_v: int,
        max_seq_len: int = 64,
        use_positional_encoding: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_k = d_k
        self.d_v = d_v
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual

        if use_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(max_seq_len, d_input) * 0.02
            )

        # W_Q, W_K, W_V — no bias (standard practice)
        self.W_Q = nn.Linear(d_input, d_k, bias=False)
        self.W_K = nn.Linear(d_input, d_k, bias=False)
        self.W_V = nn.Linear(d_input, d_v, bias=False)

        self.scale = math.sqrt(d_k)

    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : Tensor, shape (B, N, d_input)

        Returns
        -------
        Z_post : (B, N, d_v)    — attention output WITH residual (used for prediction)
        Z_pre  : (B, N, d_v)    — pure attention output BEFORE residual (for Fig 5b)
        A      : (B, N, N)      — attention weights
        Q      : (B, N, d_k)
        K      : (B, N, d_k)
        V      : (B, N, d_v)
        """
        B, N, D = X.shape

        # Positional encoding
        if self.use_positional_encoding:
            X_tilde = X + self.positional_encoding[:N].unsqueeze(0)
        else:
            X_tilde = X

        Q = self.W_Q(X_tilde)          # (B, N, d_k)
        K = self.W_K(X_tilde)          # (B, N, d_k)
        V = self.W_V(X_tilde)          # (B, N, d_v)

        # Scaled dot-product scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale   # (B, N, N)

        # Causal mask: block positions j > i
        causal_mask = torch.triu(
            torch.ones(N, N, device=X.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        A = F.softmax(scores, dim=-1)   # (B, N, N)

        # Z_pre = pure attention output (BEFORE residual)
        Z_pre = torch.bmm(A, V)        # (B, N, d_v)

        # Z_post = attention output WITH residual
        if self.use_residual and self.d_v == self.d_input:
            Z_post = Z_pre + X_tilde
        else:
            Z_post = Z_pre

        return Z_post, Z_pre, A, Q, K, V
