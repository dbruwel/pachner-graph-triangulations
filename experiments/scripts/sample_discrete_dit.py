"""Sample from a trained DiscreteDiT model.

Loads saved parameters, runs the reverse diffusion process with
discrete posterior sampling + Gumbel noise + Hungarian matching,
and reconstructs triangulations from the generated gluing matrices.
"""

import logging
import pathlib
import pickle
import numpy as np
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment

from pachner_traversal.glue_encoding import encode, gluing_to_tri
from pachner_traversal.dit_discrete import DiscreteDiT

logger = logging.getLogger(__name__)


# ── Diffusion schedule ──────────────────────────────────────────────────


def get_beta_schedule(T: int, s: float = 0.008) -> np.ndarray:
    """Cosine schedule (Nichol & Dhariwal 2021), clipped."""
    steps = np.arange(T + 1, dtype=np.float64)
    t = steps / T
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 1e-5, 0.999)
    return betas.astype(np.float32)


def get_schedule_arrays(T: int):
    betas = get_beta_schedule(T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas).astype(np.float32)
    beta_bar = (1.0 - alpha_bar).astype(np.float32)
    return betas, alpha_bar, beta_bar


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
    beta_bar_t: float,
    beta_bar_t_prev: float,
    n_nodes: int,
) -> np.ndarray:
    """Compute discrete posterior log-probabilities for x_{t-1}.

    P(x_{t-1} | x_t, x0_hat) ∝ Q_t(x_t | x_{t-1}) * Q_bar_{t-1}(x_{t-1} | x0_hat)
    """
    alpha_bar_t = 1.0 - beta_bar_t
    alpha_bar_t_prev = 1.0 - beta_bar_t_prev
    beta_t = 1.0 - alpha_bar_t / max(alpha_bar_t_prev, 1e-8)
    beta_t = np.clip(beta_t, 0.0, 1.0)

    uniform = 1.0 / n_nodes

    likelihood = (1.0 - beta_t) * x_t + beta_t * uniform
    prior = (1.0 - beta_bar_t_prev) * x0_pred + beta_bar_t_prev * uniform

    log_posterior = np.log(np.clip(likelihood * prior, 1e-12, None))

    np.fill_diagonal(log_posterior, -1e9)
    log_posterior = (log_posterior + log_posterior.T) / 2

    return log_posterior


# ── Generation ──────────────────────────────────────────────────────────


def generate(
    params,
    n_tet: int,
    n_samples: int = 1,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout_rate: float = 0.0,
    proj_dim: int = 256,
    mlp_hidden_dim: int = 256,
    mlp_num_layers: int = 2,
    T: int = 1000,
    seed: int = 42,
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
    _, alpha_bar, beta_bar = get_schedule_arrays(T)

    rng = np.random.default_rng(seed)

    model = DiscreteDiT(
        n_tet=n_tet,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        proj_dim=proj_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_num_layers=mlp_num_layers,
    )

    # Start from uniformly random matchings (fully noisy)
    x_t = np.stack([random_matching(n_nodes, rng) for _ in range(n_samples)], axis=0)

    for t_idx in range(T - 1, 0, -1):
        # Encode noisy x_t with spectral positional encoding
        x_t_encoded = encode_gluing_batch(x_t, n_tet)
        x_t_encoded_j = jnp.array(x_t_encoded)
        timesteps_j = jnp.full((n_samples,), t_idx, dtype=jnp.int32)

        # Model predicts x0 (doubly-stochastic)
        logits, x0_pred = model.apply(
            {"params": params}, x_t_encoded_j, timesteps_j, training=False
        )
        x0_pred_np = np.array(x0_pred)

        # Posterior sampling for each sample
        bbt = float(beta_bar[t_idx])
        bbt_prev = float(beta_bar[t_idx - 1])

        for i in range(n_samples):
            log_post = compute_posterior_logits(
                x_t[i], x0_pred_np[i], bbt, bbt_prev, n_nodes
            )
            x_t[i] = gumbel_soft_to_hard_matching(log_post, rng, gumbel_temperature)

        if (T - t_idx) % 100 == 0:
            logger.info(f"Sampling step {T - t_idx}/{T}")

    # Final prediction at t=1 → t=0: predict and solve assignment (no Gumbel)
    x_t_encoded = encode_gluing_batch(x_t, n_tet)
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
    from pachner_traversal.utils import results_path

    N_TET = 13
    N_SAMPLES = 10

    model_path = results_path(f"discrete_dit_models/sphere_{N_TET}tet")
    params_file = model_path / "params_final.pkl"

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
