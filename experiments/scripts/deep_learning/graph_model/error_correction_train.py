""" """

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
from pachner_traversal.glue_encoding import (
    tri_to_gluing,
    jax_encode,
    get_face_graph,
    get_vertex_graph,
)
from pachner_traversal.graphomer import EdgeReconstructionGraphomer

logger = logging.getLogger(__name__)
print(jax.devices())


class DiscreteTrainState(train_state.TrainState):
    dropout_key: jax.Array


# -- Forward process -----------------------------------------------------


def q_sample(
    x0: np.ndarray,
    n_remove: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt gluing matrices by removing edges

    Args:
        x0: [B, N, N] clean gluing matrices (binary symmetric permutation).
        n_remove: [B] number of edges to remove.
        rng: numpy random Generator for reproducibility.

    Returns:
        x_t: [B, N, N] corrupted gluing matrices (still binary permutations).
        x_removed: [B, N, N] the edges that were removed.
    """
    B = x0.shape[0]
    x_t = np.empty_like(x0)
    x_removed = np.empty_like(x0)

    for b in range(B):
        r = n_remove[b]
        if r == 0:
            x_t[b] = x0[b]
            x_removed[b] = np.zeros_like(x0[b])
            continue

        remove_ids = rng.choice(x0[b].shape[0], size=r, replace=False)
        x_working = x0[b].copy()
        x_working[remove_ids, :] = 0
        x_working[:, remove_ids] = 0
        x_removed[b] = x0[b] - x_working
        x_t[b] = x_working

    return x_t, x_removed


# -- Encoding helpers ----------------------------------------------------


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


# -- Loss ----------------------------------------------------------------


def loss_fn(
    params,
    model,
    x_removed,
    x_t_encoded,
    dist_matrix,
    dropout_key=None,
    training=True,
):
    """Cross-entropy loss on the upper triangle (i < j) only.

    The gluing matrix is symmetric with zero diagonal, so we only need
    to supervise the upper triangle to avoid double-counting.

    Args:
        params: model parameters.
        model: DiscreteDiT instance.
        x_removed: [B, N, N] edges that were removed.
        x_t_encoded: [B, N, D] positionally-encoded noisy input.
        dist_matrix: [B, N, N] distance matrices for spatial bias.
        dropout_key: PRNG key for dropout (required when training=True).
        training: dropout mode.
    """
    rngs = {"dropout": dropout_key} if dropout_key is not None else {}
    logits, _ = model.apply(
        {"params": params}, x_t_encoded, dist_matrix, training=training, rngs=rngs
    )

    loss = optax.sigmoid_binary_cross_entropy(logits, x_removed)
    return loss.sum()


# -- Helpers ------------------------------------------------------------


def write_loss(file_path, step, loss):
    with open(file_path, "a") as f:
        f.write(f"{step},{loss}\n")


def get_shortest_path_distance_matrix(adj_matrix: jnp.ndarray) -> jnp.ndarray:
    num_nodes = adj_matrix.shape[0]

    dist_init = jnp.where(adj_matrix == 1, 1.0, jnp.inf)
    dist_init = jnp.fill_diagonal(dist_init, 0.0, inplace=False)

    def update_step(dist, k):
        col_k = dist[:, [k]]
        row_k = dist[[k], :]

        dist = jnp.minimum(dist, col_k + row_k)
        return dist, None

    dist_matrix, _ = jax.lax.scan(update_step, dist_init, jnp.arange(num_nodes))
    dist_matrix = jnp.where(jnp.isinf(dist_matrix), 0, dist_matrix)

    return dist_matrix


batched_shortest_path = jax.vmap(get_shortest_path_distance_matrix, in_axes=0)


# -- Training ------------------------------------------------------------


def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    n_tet: int,
    num_train_steps: int = 100_000,
    batch_size: int = 64,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    d_model: int = 128,
    seed: int = 0,
    remove_rate: int = 10,
):
    n_nodes = 12 * n_tet
    n_remove = np.array([remove_rate] * batch_size)
    encoding_dim = 3 * 36 * n_tet

    dataset = Dataset(data_path, num_test_samps=1000)

    model = EdgeReconstructionGraphomer(
        n_tet=n_tet,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        d_model=d_model,
        max_dist=n_nodes + 1,  # effectively no limit on distance,
    )

    key = jax.random.PRNGKey(seed)
    _, params_key, dropout_key = jax.random.split(key, 3)
    sample_x = jnp.zeros((batch_size, n_nodes, encoding_dim), dtype=jnp.float32)
    sample_dist_matrix = jnp.zeros((batch_size, n_nodes, n_nodes, 4), dtype=jnp.int32)
    params = model.init(
        {"params": params_key, "dropout": dropout_key},
        sample_x,
        sample_dist_matrix,
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
    def train_step(params, x_removed: jax.Array, x_t: jax.Array, dropout_key):
        x_t_encoded = jax_encode_gluing_batch(x_t, n_tet)
        face_graph = get_face_graph(n_tet)
        vertex_graph = get_vertex_graph(n_tet)

        mask: jax.Array = jnp.eye(x_t.shape[1], dtype=bool)

        face_graph = jnp.where(mask, 1, face_graph)
        vertex_graph = jnp.where(mask, 1, vertex_graph)

        face_dist = jnp.stack([face_graph] * batch_size, axis=0)
        vertex_dist = jnp.stack([vertex_graph] * batch_size, axis=0)
        gluing_dist: jax.Array = jnp.where(mask, 2, x_t)

        cond = (x_t > 0) | (face_dist > 0) | (vertex_dist > 0)
        global_adj_matrix = jnp.where(cond, 1, 0)

        global_dist = batched_shortest_path(global_adj_matrix)

        dist_matrix = jnp.stack(
            [face_dist, vertex_dist, gluing_dist, global_dist], axis=-1
        )
        dist_matrix = dist_matrix.astype(jnp.int32)

        l, g = jax.value_and_grad(loss_fn)(
            params,
            model,
            x_removed,
            x_t_encoded,
            dist_matrix,
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

        # Forward process: corrupt the gluing matrix by removing edges
        x_t, x_removed = q_sample(x0_gluing, n_remove=n_remove, rng=rng)

        # To JAX
        x_t_j = jnp.array(x_t)
        x_removed_j = jnp.array(x_removed)

        step_dropout_key, dropout_key = jax.random.split(state.dropout_key)

        l, grads = train_step(state.params, x_removed_j, x_t_j, step_dropout_key)
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


# -- Main ----------------------------------------------------------------

if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    from pachner_traversal.utils import data_root, create_results_path

    N_TET = 13

    hdf5_path = (
        data_root
        / "input_data"
        / "dehydration"
        / "processed"
        / f"d_training_spheres_{N_TET}.hdf5"
    )
    save_path = create_results_path(f"graphomer_dev/sphere_{N_TET}tet")
    save_path.mkdir(parents=True, exist_ok=True)

    tic = time.time()
    train_model(hdf5_path, save_path, n_tet=N_TET)
    toc = time.time()
    print(f"Training time: {toc - tic:.2f} seconds")
