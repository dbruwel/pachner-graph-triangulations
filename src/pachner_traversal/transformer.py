import shutil
from functools import partial
from pathlib import Path
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.core import freeze
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import normal
from flax.training import train_state

from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.utils import get_last_csv_row, get_random_sample_idx, load_model


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


class CausalSelfAttention(nn.Module):
    # (B, L, D) -> (B, L, D)
    d_model: int
    num_heads: int
    use_mask: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        B, L, D = x.shape

        assert D % self.num_heads == 0
        head_size = D // self.num_heads

        qkv = nn.Dense(
            features=3 * D,
            name="c_attn",
            kernel_init=normal(0.02),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)

        q, k, v = jnp.split(qkv, 3, axis=-1)

        k = k.reshape(B, L, self.num_heads, head_size)
        q = q.reshape(B, L, self.num_heads, head_size)
        v = v.reshape(B, L, self.num_heads, head_size)

        if self.use_mask:
            mask_input = q[:, :, 0, 0]
            causal_mask = nn.make_causal_mask(mask_input)
            mask = causal_mask
        else:
            mask = None

        y = nn.dot_product_attention(q, k, v, mask=mask, broadcast_dropout=False)
        y = y.reshape(B, L, D)
        y = nn.Dense(
            features=self.d_model,
            name="c_proj",
            kernel_init=normal(0.02),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(y)

        return y


class Block(nn.Module):
    # (B, L, D) -> (B, L, D)
    d_model: int
    num_heads: int
    use_mask: bool = True
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = False) -> jax.Array:
        attn = CausalSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            use_mask=self.use_mask,
            name="attn",
        )

        x_ln = nn.LayerNorm(name="ln_1", dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        attn_output = attn(x_ln, training=training)
        x = x + attn_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x_ln = nn.LayerNorm(name="ln_2", dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)
        mlp_output = self.mlp(x_ln)

        x = x + mlp_output
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        return x

    def mlp(self, x: jax.Array) -> jax.Array:
        # (B, L, D) -> (B, L, D)
        d_ff = 4 * self.d_model
        c_fc = nn.Dense(
            features=d_ff, name="c_fc", dtype=jnp.bfloat16, param_dtype=jnp.float32
        )
        c_proj = nn.Dense(
            features=self.d_model,
            name="c_proj",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )

        x = c_fc(x)
        x = jax.nn.gelu(x, approximate=True)
        x = c_proj(x)
        return x


class Transformer(nn.Module):
    vocab_size: int
    d_model: int
    block_size: int  # Max sequence length.
    num_layers: int
    num_heads: int
    use_mask: bool = True
    output_size: int | None = None
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, idx: jax.Array, training: bool = False) -> jax.Array:
        output_size = self.output_size or self.vocab_size

        _, L = idx.shape
        assert L <= self.block_size, (
            f"Cannot forward sequence of length {L}, block size is only {self.block_size}"
        )
        # Token embedding.
        # (B, L) -> (B, L, D)
        wte = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            name="wte",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        tok_emb = wte(idx)

        # Learned positional embedding.
        # (L) -> (L, D)
        wpe = nn.Embed(
            num_embeddings=self.block_size,
            features=self.d_model,
            name="wpe",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        pos = jnp.arange(0, L)
        pos_emb = wpe(pos)

        # Combine embeddings.
        x = tok_emb + pos_emb  # Broadcasting (B, L, D) + (L, D) -> (B, L, D)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Transformer Blocks.
        # (B, L, D) -> (B, L, D)
        for i in range(self.num_layers):
            x = Block(
                d_model=self.d_model,
                num_heads=self.num_heads,
                use_mask=self.use_mask,
                name=f"h_{i}",
            )(x, training=training)

        # Final Layers.
        x = nn.LayerNorm(name="ln_f", dtype=jnp.bfloat16, param_dtype=jnp.float32)(x)

        # (B, L, D) -> (B, L, output_size)
        logits = nn.Dense(
            features=output_size,
            use_bias=False,
            name="lm_head",
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )(x)

        return logits


class ScalarTransformer(nn.Module):
    vocab_size: int
    d_model: int
    block_size: int  # Max sequence length.
    num_layers: int
    num_heads: int
    use_mask: bool = False
    output_size: int | None = None
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, idx: jax.Array, training: bool = False) -> jax.Array:
        # (B, L, D)
        x = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            block_size=self.block_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            use_mask=self.use_mask,
            output_size=self.output_size,
            dropout_rate=self.dropout_rate,
        )(idx, training=training)

        x = nn.LayerNorm(name="pre_pool_ln")(x)  # -> (B, L, D)
        logits = nn.Dense(1, name="attn_logits")(x)  # -> (B, L, 1)
        weights = nn.softmax(logits, axis=1)  # -> (B, L, 1)
        pooled = jnp.sum(x * weights, axis=1)  # -> (B, D)
        pooled = nn.LayerNorm(name="post_pool_ln")(pooled)
        pooled = nn.relu(nn.Dense(x.shape[-1], name="hidden_projection")(pooled))
        out = nn.Dense(1, name="final_projection")(pooled)  # -> (B, 1)

        return out.squeeze(-1)  # -> (B)


@jax.jit
def train_step_auto_regression(
    state: MinimalTrainState,
    batch_input: jax.Array,
    batch_labels: jax.Array,
) -> tuple[MinimalTrainState, jax.Array]:
    dropout_key, new_dropout_key = jax.random.split(state.dropout_key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_input,
            training=True,
            rngs={"dropout": dropout_key},
        )
        logits_fp32 = logits.astype(jnp.float32)

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits_fp32, labels=batch_labels
        ).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(dropout_key=new_dropout_key)

    return new_state, loss


@jax.jit
def train_step_scalar_regression(
    state: MinimalTrainState,
    batch_input: jax.Array,
    batch_labels: jax.Array,
) -> tuple[MinimalTrainState, jax.Array]:
    dropout_key, new_dropout_key = jax.random.split(state.dropout_key)

    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            batch_input,
            training=True,
            rngs={"dropout": dropout_key},
        )
        logits_fp32 = logits.astype(jnp.float32)

        loss = optax.squared_error(logits_fp32, batch_labels).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)

    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(dropout_key=new_dropout_key)

    return new_state, loss


@partial(jax.jit, static_argnames=["train_step"])
def train_sweep_steps(
    train_step,
    state: MinimalTrainState,
    batches_input: jax.Array,
    batches_labels: jax.Array,
):

    def scan_body(carry_current_state, data):
        b_input, b_label = data
        carry_new_state, loss = train_step(carry_current_state, b_input, b_label)
        return carry_new_state, loss

    final_state, losses = jax.lax.scan(
        scan_body, state, (batches_input, batches_labels)
    )

    return final_state, losses


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
        pred_fp32 = pred.astype(jnp.float32)

        pred_next_token = pred_fp32[:, i, :]
        sampled_token = jax.random.categorical(subkey, pred_next_token, axis=-1)
        samples = samples.at[:, i + 1].set(sampled_token)

        return samples, None

    samples, _ = jax.lax.scan(
        generate_step, initial_samples, xs=jnp.arange(seq_len - 1)
    )
    return samples


def init_train_state(model, params, dropout_key, learning_rate=0.0005):
    state = MinimalTrainState.create(
        params=params,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        learning_rate=learning_rate,
        m_tm1=freeze(jax.tree_util.tree_map(jnp.zeros_like, params)),
        v_tm1=freeze(jax.tree_util.tree_map(jnp.zeros_like, params)),
        t=0,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=0.01),
    )

    return state


def init_model(
    model_type,
    dataset: Dataset,
    encoder: Encoder,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    use_mask: bool = True,
    output_size: int | None = None,
):
    vocab_size = len(encoder.char_to_id)
    seq_len = dataset.max_len + 1

    key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = jax.random.split(key, 3)

    model = model_type(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        use_mask=use_mask,
        output_size=output_size,
    )

    return model, (main_key, params_key, dropout_key), (vocab_size, seq_len)


def init_params(
    model,
    params_key,
    load_path: Path,
    train_input: np.ndarray,
    batch_size: int,
    num_train_steps: int,
    sweep: int,
    resume: bool,
):
    resumed = (load_path / "params.pkl").exists() and resume
    if resumed:
        params = load_model(load_path)
        last_step = int(get_last_csv_row(load_path / "train_losses.csv")[0])
        steps = range(last_step, num_train_steps, sweep)
        meta = last_step
    else:
        # Clear path if it exists.
        if load_path.exists() and load_path.is_dir():
            shutil.rmtree(load_path)
        load_path.mkdir(parents=True, exist_ok=True)

        blank_idx = get_random_sample_idx(batch_size, len(train_input))
        blank_batch_input = train_input[blank_idx]

        key_data = {"params": params_key}
        blank_model = model.init(key_data, blank_batch_input, training=True)
        params = blank_model["params"]

        meta = sum(x.size for x in jax.tree_util.tree_leaves(params))

        steps = range(0, num_train_steps, sweep)

    return resumed, meta, steps, params
