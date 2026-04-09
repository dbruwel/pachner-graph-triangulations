import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Optional

from pachner_traversal.dit import DiT, TimestepEmbedding, DiTBlock


class PairwiseMLP(nn.Module):
    """MLP that takes pairs of node vectors and predicts edge logits.

    Given transformer output of shape [B, N, D], produces a logit matrix
    of shape [B, N, N] representing the likelihood of an edge between
    each pair of nodes.
    """

    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, x, training: bool = False):
        # x: [B, N, D]
        B, N, D = x.shape

        x_i = x[:, :, None, :]  # (B, N, 1, D)
        x_j = x[:, None, :, :]  # (B, 1, N, D)

        f_sum = x_i + x_j
        f_diff = jnp.abs(x_i - x_j)
        f_prod = x_i * x_j

        # Stack them along the feature dimension
        h = jnp.concatenate([f_sum, f_diff, f_prod], axis=-1)
        for i in range(self.num_layers):
            h = nn.Dense(self.hidden_dim, name=f"mlp_{i}")(h)
            h = nn.gelu(h)

        # [B, N, N, 1]
        logits = nn.Dense(1, name="mlp_out", kernel_init=nn.initializers.zeros)(h)
        logits = logits.squeeze(-1)  # [B, N, N]

        return logits


def sinkhorn(log_alpha, n_iters: int = 20, temperature: float = 0.05):
    """Sinkhorn normalisation to produce a doubly-stochastic matrix.

    Operates on log-space scores. Returns soft doubly-stochastic matrix.
    The gluing matrix is symmetric, so we symmetrise first, then apply
    Sinkhorn which enforces rows and columns sum to 1.

    Args:
        log_alpha: [B, N, N] log-scores (already symmetrised).
        n_iters: number of Sinkhorn iterations.
        temperature: temperature for soft assignment.
    """
    log_alpha = log_alpha / temperature

    for _ in range(n_iters):
        # Row normalisation (log-space)
        log_alpha = log_alpha - jax.nn.logsumexp(log_alpha, axis=-1, keepdims=True)
        # Column normalisation (log-space)
        log_alpha = log_alpha - jax.nn.logsumexp(log_alpha, axis=-2, keepdims=True)

    return log_alpha, jnp.exp(log_alpha)


class DiscreteDiT(nn.Module):
    """Discrete diffusion model for gluing matrices.

    Pipeline:
        1. Positionally encode the (noisy) gluing matrix via spectral encoding
           (done outside the model in the training loop).
        2. DiT transformer processes the encoded node vectors.
        3. Pairwise MLP predicts edge logits from transformer output.
        4. Sinkhorn normalisation produces a doubly-stochastic matrix.

    The model predicts the clean x0 gluing matrix given a noisy input.
    """

    n_tet: int
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    proj_dim: Optional[int] = 256
    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 2
    sinkhorn_iters: int = 20
    sinkhorn_temp: float = 0.05

    def setup(self):
        n_nodes = 12 * self.n_tet
        input_dim = 3 * 36 * self.n_tet  # 108 * n_tet

        self.backbone = DiT(
            input_dim=input_dim,
            seq_len=n_nodes,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            proj_dim=self.proj_dim,
            project_out=False,
        )
        self.pairwise_mlp = PairwiseMLP(
            hidden_dim=self.mlp_hidden_dim,
            num_layers=self.mlp_num_layers,
        )

    def __call__(
        self, x, timesteps, training: bool = False, apply_sinkhorn: bool = True
    ):
        """
        Args:
            x: [B, n_nodes, input_dim] positionally-encoded noisy gluing matrix.
            timesteps: [B] diffusion timestep indices.
            training: whether in training mode (dropout).
            apply_sinkhorn: whether to apply Sinkhorn normalisation at the end.

        Returns:
            logits_norm: [B, n_nodes, n_nodes] normalized edge logits.
            x0_pred: [B, n_nodes, n_nodes] doubly-stochastic prediction
                     (only if apply_sinkhorn=True, else same as logits).
        """
        # Transformer backbone
        h = self.backbone(x, timesteps, training=training)  # [B, N, input_dim]

        # Pairwise edge prediction
        logits = self.pairwise_mlp(h, training=training)  # [B, N, N]

        # Zero out the diagonal (a node cannot be glued to itself)
        n_nodes = 12 * self.n_tet
        diag_mask = jnp.eye(n_nodes)[None, :, :]
        logits = logits * (1.0 - diag_mask) - (1e9) * diag_mask

        if apply_sinkhorn:
            log_x0_pred, x0_pred = sinkhorn(
                logits, n_iters=self.sinkhorn_iters, temperature=self.sinkhorn_temp
            )
            # Re-symmetrise after Sinkhorn (it can drift slightly)
            x0_pred = (x0_pred + jnp.transpose(x0_pred, (0, 2, 1))) / 2
        else:
            x0_pred = jax.nn.sigmoid(logits)
            log_x0_pred = logits

        return log_x0_pred, x0_pred
