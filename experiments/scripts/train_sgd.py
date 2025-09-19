import os
import pickle

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from pachner_traversal.data_io import Dataset, Encoder
from pachner_traversal.reinforcement import train_step as reinforce_train_step
from pachner_traversal.transformer import (
    MinimalTrainState,
    Transformer,
    generate_samples,
    train_step,
)


def train_model(file_path, save_path, num_test_samps=1_000):
    batch_size = 32

    dataset = Dataset(file_path, num_test_samps)
    encoder = Encoder(dataset)

    sample_batch_str = dataset.samp_batch(batch_size)
    sample_batch, train_label = encoder.encode(sample_batch_str)

    vocab_size = len(encoder.char_to_id)
    d_model = 64  # Dimension of embeddings and model
    num_layers = 12  # Number of transformer blocks
    num_heads = 8  # Number of attention heads
    d_ff = 64  # Dimension of the feed-forward network
    seq_len = dataset.max_len + 1  # Sequence length
    learning_rate = 0.0005
    num_train_steps = 50_000

    key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = jax.random.split(key, 3)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    params = model.init({"params": params_key}, sample_batch, training=True)["params"]

    print(
        f"Model initialized. Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params))}"
    )

    state = MinimalTrainState.create(
        params=params,
        apply_fn=model.apply,
        dropout_key=dropout_key,
        learning_rate=learning_rate,
        m_tm1=flax.core.freeze(jax.tree_util.tree_map(jnp.zeros_like, params)),
        v_tm1=flax.core.freeze(jax.tree_util.tree_map(jnp.zeros_like, params)),
        t=0,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=0.01),
    )

    test_batch_input, test_batch_label = encoder.encode(dataset.test_data)

    losses = {}
    test_losses = {}

    print("\n--- Starting Training ---")
    for step in range(num_train_steps):
        batch_input, batch_label = encoder.encode(dataset.samp_batch(batch_size))
        state, loss = train_step(state, batch_input, batch_label)
        losses[step] = loss

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1:3d}/{num_train_steps}, Loss: {loss:.4f}")

            test_logits = state.apply_fn(
                {"params": state.params},
                test_batch_input,
                training=False,
            )

            test_one_hot_labels = jax.nn.one_hot(
                test_batch_label, num_classes=vocab_size
            )
            test_loss = optax.softmax_cross_entropy(
                test_logits, test_one_hot_labels
            ).mean()
            test_losses[step] = test_loss

    print("\n✅ Training finished.")

    (pd.Series(losses).astype(float)).to_csv(save_path + "_train_losses.csv")
    (pd.Series(test_losses).astype(float)).to_csv(save_path + "_test_losses.csv")

    with open(save_path + ".pkl", "wb") as file:
        pickle.dump(state.params, file)


if __name__ == "__main__":
    train_model(
        "train_data/all_5tet.hdf5",
        "saves/all_run/12block_8head_5tet",
        num_test_samps=700,
    )
    train_model("train_data/all_6tet.hdf5", "saves/all_run/12block_8head_6tet")
    train_model("train_data/all_7tet.hdf5", "saves/all_run/12block_8head_7tet")
    train_model("train_data/all_8tet.hdf5", "saves/all_run/12block_8head_8tet")
