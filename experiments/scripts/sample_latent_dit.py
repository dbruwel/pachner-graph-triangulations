import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent))
from pachner_traversal.dit import DiT


def sample_dit(
    params,
    model: DiT,
    num_steps: int = 1000,
    shape=(1, 144, 1296),
    rng_seed: int = 0,
    eta: float = 0.0,
):
    # Cosine schedule (match training)
    T = num_steps
    steps = np.arange(T + 1, dtype=np.float64)
    t = steps / T
    s = 0.008
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0, 0.999)
    alphas = 1.0 - betas
    sqrt_alphas = np.sqrt(alphas)
    sqrt_one_minus_alphas = np.sqrt(1 - alphas)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod[1:])
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod[1:])

    rng = jax.random.PRNGKey(rng_seed)
    x = jax.random.normal(rng, shape)
    for i in reversed(range(T)):
        t_arr = jnp.full((shape[0],), i, dtype=jnp.int32)
        pred_noise = model.apply({"params": params}, x, t_arr, training=False)
        alpha = alphas[i]
        alpha_bar = alphas_cumprod[i + 1]
        beta = betas[i]
        if i > 0:
            noise = jax.random.normal(rng, shape)
        else:
            noise = 0
        x = (
            1
            / np.sqrt(alpha)
            * (x - (beta / sqrt_one_minus_alphas_cumprod[i]) * pred_noise)
            + sqrt_one_minus_alphas[i] * noise
        )
    return x


if __name__ == "__main__":
    import sys
    import pathlib

    # Example usage: python sample_dit.py /path/to/params.pkl /path/to/output.npy
    params_path = sys.argv[1]
    output_path = sys.argv[2]
    # Model config (should match training)
    model = DiT(input_dim=1296, seq_len=144, d_model=256, num_layers=4, num_heads=4)
    with open(params_path, "rb") as f:
        params = pickle.load(f)
    samples = sample_dit(params, model, num_steps=1000, shape=(8, 144, 1296))
    np.save(output_path, np.array(samples))
    print(f"Saved samples to {output_path}")
