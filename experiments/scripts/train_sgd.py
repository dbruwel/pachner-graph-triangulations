import logging
import pathlib
import pickle
import time
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax.core import freeze

from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import (
    MinimalTrainState,
    Transformer,
    generate_samples,
    train_step,
)
from pachner_traversal.utils import data_path, results_path

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


def get_sample_idx(batch_size, dataset_size):
    return np.random.choice(dataset_size, size=batch_size, replace=True)


def train_model(
    file_path: pathlib.Path, save_path: pathlib.Path, num_test_samps: int = 1_000
) -> None:
    batch_size = 64

    dataset = Dataset(file_path, num_test_samps)
    encoder = Encoder(dataset)

    all_data_str = dataset.read_all_data()
    all_data_input, all_data_label = encoder.encode(all_data_str)

    test_input = all_data_input[dataset.test_idx]
    test_label = all_data_label[dataset.test_idx]

    train_idx = list(set(range(len(all_data_label))) - set(dataset.test_idx))
    train_idx.sort()

    train_input = all_data_input[train_idx]
    train_label = all_data_label[train_idx]

    sample_idx = get_sample_idx(batch_size, len(train_idx))
    sample_batch_input = train_input[sample_idx]

    vocab_size = len(encoder.char_to_id)
    d_model = 512  # Dimension of embeddings and model
    num_layers = 6  # Number of transformer blocks
    num_heads = 4  # Number of attention heads
    seq_len = dataset.max_len + 1  # Sequence length
    learning_rate = 0.0005
    num_train_steps = 1_000_000
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

    params = model.init({"params": params_key}, sample_batch_input, training=True)[
        "params"
    ]

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

    logger.info("\n--- Starting Training ---")
    for step in range(num_train_steps):
        sample_idx = get_sample_idx(batch_size, len(train_idx))
        batch_input = train_input[sample_idx]
        batch_label = train_label[sample_idx]

        state, loss = train_step(state, batch_input, batch_label)

        if (step + 1) % 500 == 0 or (step + 1) == num_train_steps:
            msg = f"Step {step + 1:3d}/{num_train_steps}, Loss: {float(loss):.4f}"
            logger.info(msg)

            test_loss = get_test_loss(
                state,
                test_input,
                test_label,
                vocab_size,
            )

            write_loss(save_path / "train_losses.csv", step + 1, float(loss))
            write_loss(save_path / "test_losses.csv", step + 1, float(test_loss))

            with open(save_path / "params.pkl", "wb") as file:
                pickle.dump(state.params, file)

    logger.info("\n Training finished.")

    with open(save_path / "params.pkl", "wb") as file:
        pickle.dump(state.params, file)


def sample_model(
    file_path: pathlib.Path,
    save_path: pathlib.Path,
    num_test_samps: int = 1_000,
    gen_its: int = 10,
    samps_to_gen: int = 1_000,
) -> None:
    dataset = Dataset(file_path, num_test_samps)
    encoder = Encoder(dataset)

    vocab_size = len(encoder.char_to_id)
    d_model = 512  # Dimension of embeddings and model
    num_layers = 6  # Number of transformer blocks
    num_heads = 4  # Number of attention heads
    seq_len = dataset.max_len + 1  # Sequence length
    learning_rate = 0.0005

    key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = jax.random.split(key, 3)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    with open(save_path / "params.pkl", "rb") as file:
        params = pickle.load(file)

    state = MinimalTrainState.create(
        params=params,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        learning_rate=learning_rate,
        m_tm1=flax.core.freeze(  # type: ignore
            jax.tree_util.tree_map(jnp.zeros_like, params)
        ),
        v_tm1=flax.core.freeze(  # type: ignore
            jax.tree_util.tree_map(jnp.zeros_like, params)
        ),
        t=0,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=0.01),
    )

    bos_id = encoder.char_to_id["[BOS]"]
    subkey = jax.random.PRNGKey(42)

    samps_str = []
    for i in range(gen_its):
        logger.info(f"Generating samples... Iteration {i + 1}/{gen_its}")
        subkey = jax.random.split(subkey, 1)[0]
        samps = generate_samples(state, samps_to_gen, seq_len, subkey, bos_id)
        samps_str = samps_str + encoder.decode(np.array(samps))

    with open(save_path / "generated_samples.txt", "w") as f:
        for samp in samps_str:
            f.write(samp + "\n")


if __name__ == "__main__":
    train = True
    sample = True

    logging.basicConfig(level=logging.INFO)

    Ns = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    for N in Ns:
        logger.info(f"\n\n=== N_TET = {N} ===")
        processed_data_path = data_path / "input_data" / "dehydration" / "processed"
        file_path = processed_data_path / f"d_training_spheres_{N}.hdf5"

        save_path = data_path

        if train:
            save_path = results_path(
                f"sgd_models_dehydration/output/spheres_512emb_6block_4head_{N}tet"
            )
            set_path = True

            tic = time.time()
            train_model(file_path, save_path)
            toc = time.time()

            print(f"Training time: {toc - tic:.2f} seconds")

        if sample:
            tic = time.time()
            sample_model(file_path, save_path, samps_to_gen=1000, gen_its=100)
            toc = time.time()

            print(f"Sampling time: {toc - tic:.2f} seconds")
