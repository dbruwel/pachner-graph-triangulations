import logging
import pathlib
import pickle
from functools import partial

import flax
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from flax.core import freeze

from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import MinimalTrainState, Transformer, train_step
from pachner_traversal.utils import results_path, data_path

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=["vocab_size"])
def get_test_loss(
    state: MinimalTrainState,
    test_batch_input: jax.Array,
    test_batch_label: jax.Array,
    vocab_size: int,
) -> jax.Array:
    test_logits = state.apply_fn(
        {"params": state.params},
        test_batch_input,
        training=False,
    )

    test_one_hot_labels = jax.nn.one_hot(test_batch_label, num_classes=vocab_size)
    test_loss = optax.softmax_cross_entropy(test_logits, test_one_hot_labels).mean()
    return test_loss


def write_loss(file_path, step, loss):
    with open(file_path, "a") as f:
        f.write(f"{step},{loss}\n")


def train_model(
    file_path: pathlib.Path, save_path: pathlib.Path, num_test_samps: int = 1_000
) -> None:
    batch_size = 32

    dataset = Dataset(file_path, num_test_samps)
    encoder = Encoder(dataset)

    sample_batch_str = dataset.samp_batch(batch_size)
    sample_batch, _ = encoder.encode(sample_batch_str)

    vocab_size = len(encoder.char_to_id)
    d_model = 64  # Dimension of embeddings and model
    num_layers = 4  # Number of transformer blocks
    num_heads = 4  # Number of attention heads
    seq_len = dataset.max_len + 1  # Sequence length
    learning_rate = 0.0005
    num_train_steps = 1_050
    dropout_rate = 0.1

    key = jax.random.PRNGKey(0)
    _, params_key, dropout_key = jax.random.split(key, 3)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
    )

    params = model.init({"params": params_key}, sample_batch, training=True)["params"]

    logger.info(
        f"Model initialized. Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}"
    )

    state = MinimalTrainState.create(
        params=params,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        learning_rate=learning_rate,
        m_tm1=freeze(jax.tree_util.tree_map(jnp.zeros_like, params)),
        v_tm1=freeze(jax.tree_util.tree_map(jnp.zeros_like, params)),
        t=0,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=0.01, b2=0.99),
    )

    test_batch_input, test_batch_label = encoder.encode(dataset.test_data)

    logger.info("\n--- Starting Training ---")
    for step in range(num_train_steps):
        batch_input, batch_label = encoder.encode(dataset.samp_batch(batch_size))
        state, loss = train_step(state, batch_input, batch_label)

        if (step + 1) % 500 == 0 or (step + 1) == num_train_steps:
            msg = f"Step {step + 1:3d}/{num_train_steps}, Loss: {float(loss):.4f}"
            logger.info(msg)

            test_loss = get_test_loss(
                state,
                test_batch_input,
                test_batch_label,
                vocab_size,
            )

            write_loss(save_path / "train_losses.csv", step + 1, float(loss))
            write_loss(save_path / "test_losses.csv", step + 1, float(test_loss))

            with open(save_path / "params.pkl", "wb") as file:
                pickle.dump(state.params, file)

    logger.info("\n Training finished.")

    with open(save_path / "params.pkl", "wb") as file:
        pickle.dump(state.params, file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processed_data_path = data_path / "input_data" / "dehydration" / "processed"

    file_path = processed_data_path / "d_training_spheres_13.hdf5"
    save_path = results_path("sgd_models_dehydration/spheres_256emb_6block_8head_13tet")

    train_model(file_path, save_path)
