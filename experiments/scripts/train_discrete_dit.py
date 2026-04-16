"""Discrete diffusion training and sampling for gluing matrices.

The forward process corrupts a symmetric permutation matrix (the gluing)
by applying random transpositions (pair swaps).  A transposition picks two
matched pairs (a,b) and (c,d) and re-wires them to (a,d) and (b,c).

The number of transpositions applied scales with the noise level:
    n_swaps(t) = round(beta_bar_t * n_pairs)
where n_pairs = N/2 is the total number of matched pairs and
beta_bar_t = 1 - prod_{s=1}^{t} (1 - beta_s).

At t=T the matching is essentially a uniformly random perfect matching.

The model predicts x_0 given x_t and t (x0-parameterisation).
Loss is binary cross-entropy between predicted x0 and true x0.

At generation time we predict x0_hat, then re-corrupt it to level t-1.
"""

import logging
import pathlib
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from regina import Triangulation3

from pachner_traversal.data_io_dehydration import Dataset
from pachner_traversal.glue_encoding import encode, tri_to_gluing, jax_encode
from pachner_traversal.dit_discrete import DiscreteDiT

logger = logging.getLogger(__name__)
print(jax.devices())


class DiscreteTrainState(train_state.TrainState):
    dropout_key: jax.Array


# ── Diffusion schedule ──────────────────────────────────────────────────


def get_swap_rate(n_nodes: int, T: int) -> np.ndarray:
    """Compute the swap rate at each time step."""
    alpha = (np.pi / 2 - np.pow(n_nodes, -0.5)) / T
    swap_rate = -n_nodes * np.log(np.cos(alpha * np.arange(T)))

    return swap_rate


# ── Forward process ─────────────────────────────────────────────────────


def apply_transpositions(
    perm_matrix: np.ndarray, n_swaps: int, rng: np.random.Generator
) -> np.ndarray:
    """Apply n_swaps random transpositions to a symmetric permutation matrix.

    A transposition picks two distinct matched pairs (a,b) and (c,d) and
    rewires them to (a,d) and (b,c).  The result is still a valid
    symmetric fixed-point-free involution (perfect matching).
    """
    for _ in range(n_swaps):
        order = np.arange(perm_matrix.shape[0])
        s_ids = rng.choice(perm_matrix.shape[0], size=2, replace=False)
        order[s_ids[0]], order[s_ids[1]] = (order[s_ids[1]], order[s_ids[0]])
        perm_matrix = perm_matrix[order, :][:, order]
    return perm_matrix


def q_sample(
    x0: np.ndarray,
    swap_rate_t: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Corrupt gluing matrices by applying random transpositions.

    For each sample in the batch, applies k = round(swap_rate_t * n_pairs)

    Args:
        x0: [B, N, N] clean gluing matrices (binary symmetric permutation).
        swap_rate_t: [B] swap rates.
        rng: numpy random Generator for reproducibility.

    Returns:
        x_t: [B, N, N] corrupted gluing matrices (still binary permutations).
    """
    B = x0.shape[0]
    x_t = np.empty_like(x0)

    for b in range(B):
        n_swaps = rng.poisson(swap_rate_t[b])
        x_t[b] = apply_transpositions(x0[b], n_swaps, rng)

    return x_t


# ── Encoding helpers ────────────────────────────────────────────────────


def encode_gluing_batch(gluing_matrices: np.ndarray, n_tet: int) -> np.ndarray:
    """Encode a batch of gluing matrices using spectral positional encoding.

    Args:
        gluing_matrices: [B, N, N] float gluing matrices (may be soft).
        n_tet: number of tetrahedra.

    Returns:
        encodings: [B, N, encoding_dim] float32 positional encodings.
    """
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


def sigs_to_gluings(sigs: list[str]) -> np.ndarray:
    """Convert iso-signatures to gluing matrices."""
    matrices = []
    for s in sigs:
        t = Triangulation3.rehydrate(s)
        matrices.append(tri_to_gluing(t).astype(np.float32))
    return np.stack(matrices, axis=0)


# ── Loss ────────────────────────────────────────────────────────────────


def loss_fn(
    params,
    model,
    x0_gluing,
    x_t_encoded,
    timesteps,
    n_nodes,
    dropout_key=None,
    training=True,
):
    """Cross-entropy loss on the upper triangle (i < j) only.

    The gluing matrix is symmetric with zero diagonal, so we only need
    to supervise the upper triangle to avoid double-counting.

    Args:
        params: model parameters.
        model: DiscreteDiT instance.
        x0_gluing: [B, N, N] true clean gluing matrices.
        x_t_encoded: [B, N, D] positionally-encoded noisy input.
        timesteps: [B] integer timestep indices.
        n_nodes: N = 12 * n_tet.
        dropout_key: PRNG key for dropout (required when training=True).
        training: dropout mode.
    """
    rngs = {"dropout": dropout_key} if dropout_key is not None else {}
    log_x0_pred, _ = model.apply(
        {"params": params}, x_t_encoded, timesteps, training=training, rngs=rngs
    )

    upper_mask = jnp.triu(jnp.ones((n_nodes, n_nodes)), k=1)

    loss_matrix = -(x0_gluing * log_x0_pred)

    loss = jnp.sum(loss_matrix * upper_mask) / (n_nodes / 2)
    return loss


# ── Helpers ────────────────────────────────────────────────────────────


def write_loss(file_path, step, loss):
    with open(file_path, "a") as f:
        f.write(f"{step},{loss}\n")


# ── Training ────────────────────────────────────────────────────────────


def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    n_tet: int,
    num_train_steps: int = 100_000,
    batch_size: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    proj_dim: int | None = 128,
    mlp_hidden_dim: int = 128,
    mlp_num_layers: int = 2,
    T: int = 100,
    seed: int = 0,
):
    n_nodes = 12 * n_tet
    encoding_dim = 3 * 36 * n_tet  # 108 * n_tet

    dataset = Dataset(data_path, num_test_samps=1000)
    swap_rate = get_swap_rate(n_nodes, T)

    model = DiscreteDiT(
        n_tet=n_tet,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        proj_dim=proj_dim,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_num_layers=mlp_num_layers,
    )

    key = jax.random.PRNGKey(seed)
    _, params_key, dropout_key = jax.random.split(key, 3)
    sample_x = jnp.zeros((batch_size, n_nodes, encoding_dim), dtype=jnp.float32)
    sample_t = jnp.zeros((batch_size,), dtype=jnp.int32)
    params = model.init(
        {"params": params_key, "dropout": dropout_key},
        sample_x,
        sample_t,
        training=True,
    )["params"]

    state = DiscreteTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(learning_rate=2e-4, weight_decay=0.01),
        dropout_key=dropout_key,
    )

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info(f"DiscreteDiT initialised. Parameters: {n_params:,}")

    rng = np.random.default_rng(seed)

    @jax.jit
    def train_step(params, x0_gluing, x_t, timesteps, dropout_key):
        x_t_encoded = jax_encode_gluing_batch(x_t, n_tet)
        l, g = jax.value_and_grad(loss_fn)(
            params,
            model,
            x0_gluing,
            x_t_encoded,
            timesteps,
            n_nodes,
            dropout_key=dropout_key,
            training=True,
        )
        return l, g

    for step in range(num_train_steps):
        # Sample batch of iso-signatures → gluing matrices
        idx = dataset.samp_batch_idx(batch_size)
        if len(idx) < batch_size:
            missing = batch_size - len(idx)
            new_idx = np.random.choice(idx, size=missing, replace=True)
            idx = np.concatenate([idx, new_idx])
            idx = np.sort(idx)
        sigs = dataset.read_lines(idx)
        x0_gluing = sigs_to_gluings(sigs)  # [B, N, N]

        # Sample timesteps
        timesteps_np = np.random.randint(0, T, size=(batch_size,)).astype(np.int32)
        srt = swap_rate[timesteps_np]  # [B]

        # Forward process: corrupt the gluing matrix with transpositions
        x_t = q_sample(x0_gluing, srt, rng)

        # To JAX
        x0_gluing_j = jnp.array(x0_gluing)
        x_t_j = jnp.array(x_t)
        # x_t_encoded_j = jnp.array(x_t_encoded)
        timesteps_j = jnp.array(timesteps_np)

        step_dropout_key, dropout_key = jax.random.split(state.dropout_key)
        l, grads = train_step(
            state.params, x0_gluing_j, x_t_j, timesteps_j, step_dropout_key
        )
        state = state.apply_gradients(grads=grads)
        state = state.replace(dropout_key=dropout_key)

        if (step + 1) % 500 == 0 or (step + 1) == num_train_steps:
            logger.info(f"Step {step+1}/{num_train_steps}, Loss: {float(l):.6f}")

            write_loss(save_path / "train_losses.csv", step + 1, float(l))

            with open(save_path / "params.pkl", "wb") as f:
                pickle.dump(state.params, f)

    logger.info("Training finished.")
    with open(save_path / "params_final.pkl", "wb") as f:
        pickle.dump(state.params, f)


# ── Main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    from pachner_traversal.utils import data_path, results_path

    N_TET = 13

    hdf5_path = (
        data_path
        / "input_data"
        / "dehydration"
        / "processed"
        / f"d_training_spheres_{N_TET}.hdf5"
    )
    save_path = results_path(f"discrete_dit_models/sphere_{N_TET}tet")
    save_path.mkdir(parents=True, exist_ok=True)

    tic = time.time()
    train_model(hdf5_path, save_path, n_tet=N_TET)
    toc = time.time()
    print(f"Training time: {toc - tic:.2f} seconds")
