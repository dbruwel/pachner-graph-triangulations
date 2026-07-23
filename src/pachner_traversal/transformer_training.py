import shutil
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import traverse_util
from flax.training import train_state

from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.utils import get_random_sample_idx


class MinimalTrainState(train_state.TrainState):
    dropout_key: jax.Array


@dataclass
class BaseConfig:
    dname: str
    data_path: Path
    save_path: Path
    d_model: int
    num_layers: int
    num_heads: int
    batch_size: int
    epochs: int
    num_test_samps: int | None
    num_train_steps: int | None
    sweep: int
    learning_rate: float
    use_mup: bool
    base_d_model: int
    intrem_train_loss: bool
    intrem_test_loss: bool
    final_test_loss: bool
    final_save_model: bool
    nci: bool
    flops: int | None

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


def create_get_test_loss(loss_metric_fn: Callable) -> Callable:
    @partial(jax.jit)
    def get_test_loss(
        state: MinimalTrainState,
        test_batch_input: jax.Array,
        test_batch_label: jax.Array,
    ) -> jax.Array:
        pred = state.apply_fn(
            {"params": state.params},
            test_batch_input,
            training=False,
        )
        pred_fp32 = pred.astype(jnp.float32)

        test_loss = loss_metric_fn(pred_fp32, test_batch_label).mean()
        return test_loss

    return get_test_loss


def create_train_step(loss_metric_fn: Callable) -> Callable:
    @jax.jit
    def train_step(
        state: MinimalTrainState,
        batch_input: jax.Array,
        batch_labels: jax.Array,
        num_microbatches: int = 16,
    ) -> tuple[MinimalTrainState, jax.Array]:

        mb_input = batch_input.reshape(num_microbatches, -1, *batch_input.shape[1:])
        mb_labels = batch_labels.reshape(num_microbatches, -1, *batch_labels.shape[1:])

        def microbatch_step(carry, xs):
            key, grad_acc, loss_acc = carry
            x, y = xs

            key, subkey = jax.random.split(key)

            def loss_fn(params):
                preds = state.apply_fn(
                    {"params": params},
                    x,
                    training=True,
                    rngs={"dropout": subkey},
                )
                preds_fp32 = preds.astype(jnp.float32)

                all_loss = loss_metric_fn(preds_fp32, y)
                loss = all_loss.mean() / num_microbatches
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(state.params)

            grad_acc = jax.tree_util.tree_map(lambda a, b: a + b, grad_acc, grads)
            loss_acc += loss

            return (key, grad_acc, loss_acc), None

        init_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        init_carry = (state.dropout_key, init_grads, jnp.array(0.0))

        (new_dropout_key, total_grads, total_loss), _ = jax.lax.scan(
            microbatch_step, init_carry, (mb_input, mb_labels)
        )

        new_state = state.apply_gradients(grads=total_grads)
        new_state = new_state.replace(dropout_key=new_dropout_key)

        return new_state, total_loss

    return train_step


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


def create_mup_optimizer(
    base_schedule, d_model: int, base_d_model: int = 64, weight_decay: float = 0.01
):
    width_mult = d_model / base_d_model

    def scaled_schedule(step):
        return base_schedule(step) / width_mult

    optimizers = {
        "standard": optax.adamw(learning_rate=base_schedule, weight_decay=weight_decay),
        "scaled": optax.adamw(learning_rate=scaled_schedule, weight_decay=weight_decay),
    }

    def map_params_to_optimizer(params):
        flat_params = traverse_util.flatten_dict(params)
        label_dict = {}
        for path, param in flat_params.items():
            path_str = "/".join(path)
            if (
                "wte" in path_str
                or "wpe" in path_str
                or "ln" in path_str
                or "bias" in path_str
            ):
                label_dict[path] = "standard"
            else:
                label_dict[path] = "scaled"
        return traverse_util.unflatten_dict(label_dict)

    return optax.multi_transform(optimizers, map_params_to_optimizer)  # type: ignore


def init_train_state(
    model,
    params,
    dropout_key,
    train_steps=1e6,
    peak_learning_rate=0.0005,
    final_learning_rate_frac=0.1,
    warmup_frac=0.05,
    d_model=64,
    base_d_model=64,
    weight_decay=0.01,
    use_mup: bool = False,
):
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_learning_rate,
        warmup_steps=int(train_steps * warmup_frac),
        decay_steps=int(train_steps * (1 - warmup_frac)),
        end_value=peak_learning_rate * final_learning_rate_frac,
    )

    if use_mup:
        tx = create_mup_optimizer(
            base_schedule=lr_schedule,
            d_model=d_model,
            base_d_model=base_d_model,
            weight_decay=weight_decay,
        )
    else:
        tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)

    state = MinimalTrainState.create(
        params=params,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        tx=tx,
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
    use_mup: bool = False,
    base_d_model: int = 64,
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
        use_mup=use_mup,
        base_d_model=base_d_model,
    )

    return model, (main_key, params_key, dropout_key), (vocab_size, seq_len)


def init_params(
    model,
    params_key,
    load_path: Path,
    dataset: Dataset,
    encoder: Encoder,
    batch_size: int,
    sweep: int,
    num_train_steps: int | None = None,
    flops: float | None = None,
    seq_len: int | None = None,
):
    # Clear path if it exists.
    if load_path.exists() and load_path.is_dir():
        shutil.rmtree(load_path)
    load_path.mkdir(parents=True, exist_ok=True)

    blank_idx = get_random_sample_idx(batch_size, len(dataset))
    blank_data = dataset.read_lines(blank_idx)
    blank_batch_input, _ = encoder.encode(blank_data)

    key_data = {"params": params_key}
    blank_model = model.init(key_data, blank_batch_input, training=True)
    params = blank_model["params"]

    model_size = sum(x.size for x in jax.tree_util.tree_leaves(params))

    if num_train_steps is None:
        if flops is None or seq_len is None:
            raise TypeError("Must specify `num_train_steps` or `flops` and `seq_len`")
        else:
            token_count = flops / 6 / model_size
            num_train_steps = int(token_count / seq_len / batch_size)

    steps = range(0, num_train_steps, sweep)

    return model_size, steps, params, num_train_steps
