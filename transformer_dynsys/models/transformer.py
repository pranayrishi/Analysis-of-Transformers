"""
Single-layer, single-head Transformer for dynamical systems (Figure 1).

Architecture:
  Input → [Linear Embedding] → Causal Self-Attention → [Linear | MLP] → ŷ

Only the *last* token z_n is fed to the output head (autoregressive setup).
"""

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class InputEmbedding(nn.Module):
    """Learned linear embedding  x^{emb}(t) = x(t) W^{emb}  (Section 4.1)."""

    def __init__(self, d_obs: int, d_inner: int):
        super().__init__()
        self.projection = (
            nn.Linear(d_obs, d_inner, bias=False)
            if d_obs != d_inner
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SingleLayerTransformer(nn.Module):
    """
    Parameters
    ----------
    d_obs : int
        Raw observation dimension per time step.
    d_inner : int
        Internal / latent (embedding) dimension.
    d_k, d_v : int | None
        Query/key and value dimensions (default: d_inner).
    d_output : int | None
        Output prediction dimension (default: d_obs).
    seq_len : int
        Number of delay tokens in the input window.
    use_mlp : bool
        True → MLP output head (Eq. 5); False → linear (Eq. 6).
    mlp_hidden : int
        Hidden-layer width of the MLP head.
    mlp_activation : str
        ``'relu'`` or ``'tanh'``.
    use_positional_encoding : bool
        Passed through to the attention layer.
    use_residual : bool
        Passed through to the attention layer.
    """

    def __init__(
        self,
        d_obs: int,
        d_inner: int,
        d_k: int | None = None,
        d_v: int | None = None,
        d_output: int | None = None,
        seq_len: int = 5,
        use_mlp: bool = True,
        mlp_hidden: int = 64,
        mlp_activation: str = "relu",
        use_positional_encoding: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        d_k = d_k or d_inner
        d_v = d_v or d_inner
        d_output = d_output or d_obs

        self.d_obs = d_obs
        self.d_inner = d_inner
        self.d_output = d_output
        self.seq_len = seq_len
        self.use_mlp = use_mlp
        self.use_residual = use_residual

        # Input embedding
        self.embedding = InputEmbedding(d_obs, d_inner)

        # Attention
        self.attention = CausalSelfAttention(
            d_input=d_inner,
            d_k=d_k,
            d_v=d_v,
            max_seq_len=seq_len,
            use_positional_encoding=use_positional_encoding,
            use_residual=use_residual,
        )

        # Output head -------------------------------------------------
        # Effective input dim to the head:
        #   when residual is on and d_v == d_inner  →  d_inner
        #   otherwise                                →  d_v
        head_input_dim = d_inner if (use_residual and d_v == d_inner) else d_v

        if use_mlp:
            act = nn.ReLU() if mlp_activation == "relu" else nn.Tanh()
            self.output_head = nn.Sequential(
                nn.Linear(head_input_dim, mlp_hidden),
                act,
                nn.Linear(mlp_hidden, d_output),
            )
        else:
            self.output_head = nn.Linear(head_input_dim, d_output, bias=False)

    # -----------------------------------------------------------------
    def forward(self, X: torch.Tensor):
        """
        Parameters
        ----------
        X : (B, seq_len, d_obs)

        Returns
        -------
        y_pred   : (B, d_output)
        internals: dict with keys:
            Z          — pure attention output BEFORE residual (for Fig 5b)
            Z_residual — attention + residual (for prediction & Fig 5c/5d)
            A, Q, K, V, X_emb, z_n
        """
        X_emb = self.embedding(X)                               # (B, N, d_inner)
        Z_post, Z_pre, A, Q, K, V = self.attention(X_emb)      # post=residual, pre=pure

        z_n = Z_post[:, -1, :]                                  # last token (post-residual)
        y_pred = self.output_head(z_n)                           # (B, d_output)

        return y_pred, dict(
            Z=Z_pre,              # pure attention (before residual)
            Z_residual=Z_post,    # attention + residual
            A=A, Q=Q, K=K, V=V,
            X_emb=X_emb, z_n=z_n,
        )
