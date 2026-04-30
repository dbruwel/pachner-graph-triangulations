import logging
import pathlib
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze
from flax.training import train_state
from regina import Triangulation3
from pachner_traversal.data_io_dehydration import Dataset
from pachner_traversal.glue_encoding import encode
from pachner_traversal.dit import DiT

logger = logging.getLogger(__name__)


class DiffusionTrainState(train_state.TrainState):
    dropout_key: jax.Array


# Simple linear beta schedule
def get_beta_schedule(T, s=0.008):
    # Cosine schedule from Nichol & Dhariwal 2021 (Improved DDPM)
    steps = np.arange(T + 1, dtype=np.float64)
    t = steps / T
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)
    return betas.astype(np.float32)


def q_sample(x0, t, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    # x0: [B, N, D], t: [B], noise: [B, N, D]
    return (
        sqrt_alphas_cumprod[t][:, None, None] * x0
        + sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise
    )


def loss_fn(
    params,
    state,
    model,
    batch,
    timesteps,
    noise,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    training=True,
):
    x_noisy = q_sample(
        batch, timesteps, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
    )
    pred_noise = model.apply({"params": params}, x_noisy, timesteps, training=training)
    return jnp.mean((pred_noise - noise) ** 2)


def encode_batch(sigs: list[str]) -> np.ndarray:
    """Encode a list of iso-signatures into a float32 array on the fly."""
    arrays = [encode(Triangulation3.fromIsoSig(s)).astype(np.float32) for s in sigs]
    return np.stack(arrays, axis=0)


def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    num_train_steps: int = 10000,
    batch_size: int = 16,
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    T: int = 1000,
    seed: int = 0,
):
    # Load dataset of iso-signatures from HDF5 (compact, ~85 MB)
    dataset = Dataset(data_path, num_test_samps=1000)
    N = len(dataset)
    # Infer shape from a single sample
    sample_enc = encode(Triangulation3.fromIsoSig(dataset.read_lines([0])[0])).astype(
        np.float32
    )
    seq_len, input_dim = sample_enc.shape

    # Diffusion schedule
    betas = get_beta_schedule(T)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    # Model
    model = DiT(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )
    key = jax.random.PRNGKey(seed)
    _, params_key, dropout_key = jax.random.split(key, 3)
    sample_x = jnp.zeros((batch_size, seq_len, input_dim), dtype=jnp.float32)
    sample_t = jnp.zeros((batch_size,), dtype=jnp.int32)
    params = model.init({"params": params_key}, sample_x, sample_t, training=True)[
        "params"
    ]

    state = DiffusionTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(learning_rate=2e-4, weight_decay=0.01),
        dropout_key=dropout_key,
    )

    logger.info(
        f"Model initialized. Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}"
    )

    for step in range(num_train_steps):
        # Sample batch and encode on the fly
        idx = dataset.samp_batch_idx(batch_size)
        sigs = dataset.read_lines(idx)
        batch = encode_batch(sigs)
        noise = np.random.randn(*batch.shape).astype(np.float32)
        timesteps = np.random.randint(0, T, size=(batch_size,), dtype=np.int32)
        # JAX arrays
        batch = jnp.array(batch)
        noise = jnp.array(noise)
        timesteps = jnp.array(timesteps)
        # Loss
        l = loss_fn(
            state.params,
            state,
            model,
            batch,
            timesteps,
            noise,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        )
        grads = jax.grad(loss_fn)(
            state.params,
            state,
            model,
            batch,
            timesteps,
            noise,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        )
        state = state.apply_gradients(grads=grads)
        if (step + 1) % 500 == 0 or (step + 1) == num_train_steps:
            logger.info(f"Step {step+1}/{num_train_steps}, Loss: {float(l):.4f}")
            with open(save_path / "params.pkl", "wb") as f:
                pickle.dump(state.params, f)
    logger.info("Training finished.")
    with open(save_path / "params_final.pkl", "wb") as f:
        pickle.dump(state.params, f)


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    from pachner_traversal.utils import data_path, results_path

    hdf5_path = (
        data_path
        / "input_data"
        / "dehydration"
        / "processed"
        / "d_training_spheres_13.hdf5"
    )
    save_path = results_path("dit_models/sphere_13tet")
    save_path.mkdir(parents=True, exist_ok=True)
    tic = time.time()
    train_model(hdf5_path, save_path)
    toc = time.time()
    print(f"Training time: {toc - tic:.2f} seconds")
