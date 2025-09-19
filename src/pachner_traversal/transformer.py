from functools import partial

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import matplotlib.pyplot as plt
import optax
from flax import struct
from flax.training import train_state
from flax.linen.initializers import normal
from flax.training import train_state


class MinimalTrainState(train_state.TrainState):
    params: flax.core.FrozenDict
    apply_fn: callable = struct.field(pytree_node=False)
    dropout_key: jax.random.PRNGKey
    learning_rate: float = struct.field(pytree_node=False)
    m_tm1: flax.core.FrozenDict = struct.field(pytree_node=True)
    v_tm1: flax.core.FrozenDict = struct.field(pytree_node=True)
    beta_1: float = 0.9
    beta_2: float = 0.99
    eps: float = 1e-8
    t: int = 0
    weight_decay: float = 0.01


def new_gelu(x):
    """A JAX implementation of the specific GELU activation from the PyTorch code."""
    return (
        0.5
        * x
        * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
    )


class CausalSelfAttention(nn.Module):
    """A vanilla multi-head masked self-attention layer, JAX/Flax version."""

    d_model: int  # n_embd
    num_heads: int  # n_head
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        B, T, C = (
            x.shape 
        )  # Batch size, sequence length, embedding dimensionality (d_model)

        # Ensure d_model is divisible by num_heads
        assert C % self.num_heads == 0
        head_size = C // self.num_heads

        qkv = nn.Dense(features=3 * C, name="c_attn", kernel_init=normal(0.02))(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        k = k.reshape(B, T, self.num_heads, head_size)
        q = q.reshape(B, T, self.num_heads, head_size)
        v = v.reshape(B, T, self.num_heads, head_size)

        mask_input = q[:, :, 0, 0]
        causal_mask = nn.make_causal_mask(mask_input)

        y = nn.dot_product_attention(q, k, v, mask=causal_mask, broadcast_dropout=False)
        y = y.reshape(B, T, C)
        y = nn.Dense(features=self.d_model, name="c_proj", kernel_init=normal(0.02))(y)

        return y


class Block(nn.Module):
    """An unassuming Transformer block, JAX/Flax version."""

    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = False):
        attn_output = CausalSelfAttention(
            d_model=self.d_model, num_heads=self.num_heads, name="attn"
        )(nn.LayerNorm(name="ln_1")(x), training=training)
        x = x + attn_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        mlp_output = self.mlp(nn.LayerNorm(name="ln_2")(x))
        x = x + mlp_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x

    def mlp(self, x):
        """The MLP sub-layer."""
        d_ff = 4 * self.d_model
        c_fc = nn.Dense(features=d_ff, name="c_fc")
        c_proj = nn.Dense(features=self.d_model, name="c_proj")

        x = c_fc(x)
        x = new_gelu(x)
        x = c_proj(x)
        return x


class Transformer(nn.Module):
    """Transformer Language Model, exactly as seen in GPT-2, JAX/Flax version."""

    vocab_size: int
    d_model: int  # n_embd
    block_size: int  # max sequence length
    num_layers: int  # n_layer
    num_heads: int  # n_head
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self, idx, training: bool = False
    ):  # training flag is unused but good practice
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

        # --- Input & Embeddings ---
        # Token embedding
        wte = nn.Embed(
            num_embeddings=self.vocab_size, features=self.d_model, name="wte"
        )
        tok_emb = wte(idx)

        # Learned positional embedding
        wpe = nn.Embed(
            num_embeddings=self.block_size, features=self.d_model, name="wpe"
        )
        pos = jnp.arange(0, T)
        pos_emb = wpe(pos)

        # Combine embeddings
        x = tok_emb + pos_emb  # Broadcasting (1, T, C) over (B, T, C)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # --- Transformer Blocks ---
        for i in range(self.num_layers):
            x = Block(d_model=self.d_model, num_heads=self.num_heads, name=f"h_{i}")(
                x, training=training
            )

        # --- Final Layers ---
        x = nn.LayerNorm(name="ln_f")(x)

        # Language model head
        logits = nn.Dense(features=self.vocab_size, use_bias=False, name="lm_head")(x)

        return logits


@jax.jit
def train_step(
    state: MinimalTrainState, batch_input: jnp.ndarray, batch_labels: jnp.ndarray
):
    """Performs a single training step using Optax."""
    # Split the dropout key for this step
    dropout_key, new_dropout_key = jax.random.split(state.dropout_key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_input,
            training=True,
            rngs={"dropout": dropout_key},
        )
        # Using optax's cross-entropy function is also a clean option
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_labels
        ).mean()
        return loss

    # Calculate gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    # >>> THIS IS THE KEY CHANGE <<<
    # Apply gradients to update the model parameters and optimizer state.
    # This single line replaces all your manual AdamW calculations.
    new_state = state.apply_gradients(grads=grads)

    # Update the dropout key in the new state
    new_state = new_state.replace(dropout_key=new_dropout_key)

    # Note: JAX's jit will compile this function, so returning loss and grads
    # for logging is efficient.
    return new_state, loss


def generate_samples(state, samps_to_gen, seq_len, subkey, bos_id):
    initial_samples = jnp.zeros((samps_to_gen, seq_len), dtype=jnp.int32)
    initial_samples = initial_samples.at[:, 0].set(bos_id)

    def generate_step(carry, i):
        samples = carry

        pred = state.apply_fn(
            {"params": state.params},
            samples,
            training=False,
        )

        pred_next_token = pred[:, i, :]
        sampled_token = jax.random.categorical(subkey, pred_next_token, axis=-1)
        samples = samples.at[:, i + 1].set(sampled_token)

        return samples, None

    samples, _ = jax.lax.scan(
        generate_step, initial_samples, xs=jnp.arange(seq_len - 1)
    )
    return samples
