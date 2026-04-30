"""Sample from a trained DiscreteDiT model.

Loads saved parameters, runs the reverse diffusion process with
discrete posterior sampling + Gumbel noise + Hungarian matching,
and reconstructs triangulations from the generated gluing matrices.
"""

import logging
import pathlib
import pickle

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from pachner_traversal.dit_discrete import DiscreteDiT
from pachner_traversal.glue_encoding import encode, gluing_to_tri, jax_encode

logger = logging.getLogger(__name__)


# ── Diffusion schedule ──────────────────────────────────────────────────


def get_swap_rate(n_nodes: int, T: int) -> np.ndarray:
    """Compute the swap rate at each time step."""
    alpha = (np.pi / 2 - np.pow(n_nodes, -0.5)) / T
    swap_rate = -n_nodes * np.log(np.cos(alpha * np.arange(T)))

    return swap_rate


# ── Matching helpers ────────────────────────────────────────────────────


def random_matching(n_nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a uniformly random perfect matching as a permutation matrix."""
    perm = rng.choice(n_nodes, size=n_nodes, replace=False)
    pairs = perm.reshape(-1, 2)

    match_left = np.r_[pairs[:, 0], pairs[:, 1]]
    match_right = np.r_[pairs[:, 1], pairs[:, 0]]

    matching = np.zeros((n_nodes, n_nodes), dtype=int)
    matching[match_left, match_right] = 1

    return matching


def encode_gluing_batch(gluing_matrices: np.ndarray, n_tet: int) -> np.ndarray:
    """Encode a batch of gluing matrices using spectral positional encoding."""
    B = gluing_matrices.shape[0]
    results = []
    for i in range(B):
        enc = encode(gluing_matrix=gluing_matrices[i], n_tet=n_tet).astype(np.float32)
        results.append(enc)
    return np.stack(results, axis=0)


def jax_encode_gluing_batch(gluing_matrices: jnp.ndarray, n_tet: int) -> jnp.ndarray:
    """Encode a batch of gluing matrices using spectral positional encoding.

    Args:
        gluing_matrices: [B, N, N] float gluing matrices (may be soft).
        n_tet: number of tetrahedra.

    Returns:
        encodings: [B, N, encoding_dim] float32 positional encodings.
    """
    batched_jax_encode = jax.vmap(jax_encode, in_axes=(0, None))
    results = batched_jax_encode(gluing_matrices, n_tet)
    return results


# ── Rounding helpers ────────────────────────────────────────────────────


def soft_to_hard_matching(soft_matrix: np.ndarray) -> np.ndarray:
    """Convert a soft doubly-stochastic matrix to a hard perfect matching.

    Uses the Hungarian algorithm (linear_sum_assignment) on the negative
    soft matrix to find the optimal 1-to-1 assignment.  Then symmetrises
    the result since our gluing is an undirected matching.
    """
    N = soft_matrix.shape[0]
    row_ind, col_ind = linear_sum_assignment(-soft_matrix)
    hard = np.zeros((N, N), dtype=np.float32)
    for r, c in zip(row_ind, col_ind):
        hard[r, c] = 1.0
        hard[c, r] = 1.0
    return hard


def gumbel_soft_to_hard_matching(
    log_scores: np.ndarray, rng: np.random.Generator, temperature: float = 1.0
) -> np.ndarray:
    """Add Gumbel noise to log-scores then solve the matching.

    Implements Gumbel-matching: sample a permutation from a distribution
    proportional to exp(log_scores) by adding Gumbel(0,1) noise and
    solving the resulting assignment optimally.
    """
    N = log_scores.shape[0]
    u = rng.uniform(1e-10, 1.0 - 1e-10, size=(N, N))
    gumbel_noise = -np.log(-np.log(u))
    gumbel_noise = (gumbel_noise + gumbel_noise.T) / 2
    perturbed = log_scores + temperature * gumbel_noise
    return soft_to_hard_matching(perturbed)


# ── Posterior sampling ──────────────────────────────────────────────────


def compute_posterior_logits(
    x_t: np.ndarray,
    x0_pred: np.ndarray,
    lambda_t: np.ndarray,
    lambda_t_minus_1: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Computes the log-space logits for the discrete categorical posterior.

    Args:
        x_t: [B, N, N] current noisy matrices (exact binary permutations).
        x0_pred: [B, N, N] predicted clean matrices (continuous probabilities).
        lambda_t: [B] swap rates at the current time step t.
        lambda_t_minus_1: [B] swap rates at the target time step t-1.
        eps: small constant for numerical stability.

    Returns:
        logits_t_minus_1: [B, N, N] unnormalized log probabilities.
    """
    B, N, _ = x_t.shape

    lam_t = lambda_t[:, None, None]
    lam_t_prev = lambda_t_minus_1[:, None, None]

    alpha_t_prev = np.exp(-2.0 * lam_t_prev / N)
    alpha_t_given_t_prev = np.exp(-2.0 * (lam_t - lam_t_prev) / N)

    prior = x0_pred * alpha_t_prev + (1.0 - alpha_t_prev) / N

    p_stay = alpha_t_given_t_prev + (1.0 - alpha_t_given_t_prev) / N
    p_move = 1.0 - p_stay

    likelihood = (x_t * p_stay) + ((1.0 - x_t) * p_move)

    logits_t_minus_1 = np.log(likelihood + eps) + np.log(prior + eps)

    return logits_t_minus_1


# ── Generation ──────────────────────────────────────────────────────────


def generate(
    params,
    n_tet: int,
    n_samples: int = 8,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout_rate: float = 0.0,
    proj_dim: int = 128,
    mlp_hidden_dim: int = 128,
    mlp_num_layers: int = 2,
    T: int = 100,
    seed: int = 0,
    gumbel_temperature: float = 1.0,
):
    """Generate gluing matrices by iterative denoising from pure noise.

    At each step t → t-1:
        1. Encode x_t (a binary matching) with spectral positional encoding.
        2. Model predicts x0_hat (doubly-stochastic via Sinkhorn).
        3. Compute discrete posterior logits:
           log P(x_{t-1} | x_t, x0_hat) ∝ log Q_t(x_t|x_{t-1}) + log Q_{t-1}(x_{t-1}|x0_hat)
        4. Add Gumbel noise and solve via linear_sum_assignment → binary x_{t-1}.

    Returns a list of Regina Triangulation3 objects (or None on failure).
    """
    n_nodes = 12 * n_tet
    swap_rate = get_swap_rate(n_nodes, T)

    rng = np.random.default_rng(seed)

    model = DiscreteDiT(
        n_tet=n_tet,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        proj_dim=proj_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_num_layers=mlp_num_layers,
    )

    # Start from uniformly random matchings (fully noisy)
    x_t_np = np.stack([random_matching(n_nodes, rng) for _ in range(n_samples)], axis=0)

    for t_idx in range(T - 1, 0, -1):
        # Encode noisy x_t with spectral positional encoding
        x_t_encoded = jax_encode_gluing_batch(
            jnp.array(x_t_np, dtype=np.float32), n_tet
        )
        x_t_encoded_j = jnp.array(x_t_encoded)
        timesteps_j = jnp.full((n_samples,), t_idx, dtype=jnp.int32)

        # Model predicts x0 (doubly-stochastic)
        logits, x0_pred = model.apply(
            {"params": params}, x_t_encoded_j, timesteps_j, training=False
        )
        x0_pred_np = np.array(x0_pred)

        # Posterior sampling for each sample
        srt = float(swap_rate[t_idx])
        srt_prev = float(swap_rate[t_idx - 1])

        log_post = compute_posterior_logits(
            x_t_np,
            x0_pred_np,
            np.array([srt] * n_samples),
            np.array([srt_prev] * n_samples),
        )

        for i in range(n_samples):
            x_t_np[i] = gumbel_soft_to_hard_matching(
                log_post[i], rng, gumbel_temperature
            )

        logger.info(f"Sampling step {T - t_idx}/{T}")

    # Final prediction at t=1 → t=0: predict and solve assignment (no Gumbel)
    x_t_encoded = jax_encode_gluing_batch(jnp.array(x_t_np, dtype=np.float32), n_tet)
    x_t_encoded_j = jnp.array(x_t_encoded)
    timesteps_j = jnp.zeros((n_samples,), dtype=jnp.int32)
    logits, x0_pred = model.apply(
        {"params": params}, x_t_encoded_j, timesteps_j, training=False
    )
    x0_pred_np = np.array(x0_pred)

    # Deterministic rounding at the final step
    triangulations = []
    for i in range(n_samples):
        hard = soft_to_hard_matching(x0_pred_np[i])
        try:
            tri = gluing_to_tri(hard)
            triangulations.append(tri)
        except Exception as e:
            logger.warning(f"Sample {i}: failed to reconstruct triangulation: {e}")
            triangulations.append(None)

    return triangulations


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    from pachner_traversal.utils import data_root

    N_TET = 13
    N_SAMPLES = 8

    model_path = (
        data_root / "results" / "discrete_dit_models" / "sphere_13tet" / "20260417_1100"
    )
    params_file = model_path / "params.pkl"

    with open(params_file, "rb") as f:
        params = pickle.load(f)

    logger.info(f"Loaded params from {params_file}")

    tic = time.time()
    triangulations = generate(params, n_tet=N_TET, n_samples=N_SAMPLES)
    toc = time.time()

    for i, tri in enumerate(triangulations):
        if tri is not None:
            print(f"Sample {i}: {tri.isoSig()}  (valid={tri.isValid()})")
        else:
            print(f"Sample {i}: FAILED")

    print(f"Generation time: {toc - tic:.2f} seconds")
