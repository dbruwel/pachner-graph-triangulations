from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import normal
from flax.training import train_state


class MinimalTrainState(train_state.TrainState):
    params: FrozenDict
    apply_fn: Callable = struct.field(pytree_node=False)
    dropout_key: jax.Array
    learning_rate: float = struct.field(pytree_node=False)
    m_tm1: FrozenDict = struct.field(pytree_node=True)
    v_tm1: FrozenDict = struct.field(pytree_node=True)
    beta_1: float = 0.9
    beta_2: float = 0.99
    eps: float = 1e-8
    t: int = 0
    weight_decay: float = 0.01


def gelu(x: jax.Array) -> jax.Array:
    return (
        0.5
        * x
        * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
    )


class CausalSelfAttention(nn.Module):
    d_model: int  # n_embd
    num_heads: int  # n_head
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        B, T, C = (
            x.shape
        )  # Batch size, sequence length, embedding dimensionality (d_model)

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
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        attn_output = CausalSelfAttention(
            d_model=self.d_model, num_heads=self.num_heads, name="attn"
        )(nn.LayerNorm(name="ln_1")(x), training=training)
        x = x + attn_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        mlp_output = self.mlp(nn.LayerNorm(name="ln_2")(x))
        x = x + mlp_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x

    def mlp(self, x: jax.Array) -> jax.Array:
        d_ff = 4 * self.d_model
        c_fc = nn.Dense(features=d_ff, name="c_fc")
        c_proj = nn.Dense(features=self.d_model, name="c_proj")

        x = c_fc(x)
        x = gelu(x)
        x = c_proj(x)
        return x


class Transformer(nn.Module):
    vocab_size: int
    d_model: int  # n_embd
    block_size: int  # max sequence length
    num_layers: int  # n_layer
    num_heads: int  # n_head
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, idx: jax.Array, training: bool = False) -> jax.Array:
        B, T = idx.shape
        assert (
            T <= self.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.block_size}"

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

        logits = nn.Dense(features=self.vocab_size, use_bias=False, name="lm_head")(x)

        return logits


@jax.jit
def train_step(
    state: MinimalTrainState, batch_input: jax.Array, batch_labels: jax.Array
) -> tuple[MinimalTrainState, jax.Array]:
    dropout_key, new_dropout_key = jax.random.split(state.dropout_key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_input,
            training=True,
            rngs={"dropout": dropout_key},
        )

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch_labels
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(dropout_key=new_dropout_key)

    return new_state, loss


def generate_samples(
    state: MinimalTrainState,
    samps_to_gen: int,
    seq_len: int,
    subkey: jax.Array,
    bos_id: int,
) -> jax.Array:
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
