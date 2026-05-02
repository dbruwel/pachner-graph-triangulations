import csv
import logging
import os
import pathlib
import pickle
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import freeze
from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import (
    MinimalTrainState,
    Transformer,
    generate_samples,
    train_step_auto_regression,
)
from pachner_traversal.utils import data_root as data_home
from pachner_traversal.utils import send_ntfy

logger = logging.getLogger(__name__)


# simple utility
def write_loss(save_path, step, loss):
    with open(save_path, "a") as f:
        f.write(f"{step},{loss}\n")


def save_model(save_path, state):
    with open(save_path / "params.pkl", "wb") as file:
        pickle.dump(state.params, file)


def load_model(save_path):
    with open(save_path / "params.pkl", "rb") as file:
        params = pickle.load(file)

    return params


def write_stat(stat_file_path, stat_name, stat_value):
    with open(stat_file_path, "a") as f:
        f.write(f"{stat_name}, {stat_value}\n")


def get_sample_idx(batch_size, dataset_size):
    return np.random.choice(dataset_size, size=batch_size, replace=True)


def get_last_csv_row(filepath):
    with open(filepath, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)

        last_line = f.readline().decode("utf-8")

    return next(csv.reader([last_line]))


# jax utility
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


def init_model(
    dataset: Dataset,
    encoder: Encoder,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
):
    vocab_size = len(encoder.char_to_id)
    seq_len = dataset.max_len + 1

    key = jax.random.PRNGKey(0)
    main_key, params_key, dropout_key = jax.random.split(key, 3)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        block_size=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    return model, (main_key, params_key, dropout_key), (vocab_size, seq_len)


def init_train_state(model, params, dropout_key):
    learning_rate = 0.0005

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


# critical functions
def sample_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    num_test_samps: int = 1_000,
    gen_its: int = 10,
    samps_to_gen: int = 1_000,
    tag: str | None = None,
) -> None:
    # setup model
    dataset = Dataset(data_path, num_test_samps)
    encoder = Encoder(dataset)

    params = load_model(save_path)

    model, keys, meta = init_model(
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    _, _, dropout_key = keys
    _, seq_len = meta

    state = init_train_state(model, params, dropout_key)

    # generate samples
    bos_id = encoder.char_to_id["[BOS]"]
    subkey = jax.random.PRNGKey(42)

    samps_str = []
    for i in range(gen_its):
        logger.info(f"Generating samples... Iteration {i + 1}/{gen_its}")
        subkey = jax.random.split(subkey, 1)[0]
        samps = generate_samples(state, samps_to_gen, seq_len, subkey, bos_id)
        samps_str = samps_str + encoder.decode(np.array(samps))

    # save samps
    fname = f"generated_samples_{tag}.txt" if tag else "generated_samples.txt"
    samp_save_path = save_path / fname

    with open(samp_save_path, "w") as f:
        for samp in samps_str:
            f.write(samp + "\n")


@jax.jit
def train_10k_steps(
    state: MinimalTrainState, batches_input: jax.Array, batches_labels: jax.Array
):

    def scan_body(current_state, carry):
        b_input, b_label = carry
        new_state, loss = train_step_auto_regression(current_state, b_input, b_label)
        return new_state, loss

    final_state, losses = jax.lax.scan(
        scan_body, state, (batches_input, batches_labels)
    )

    return final_state, jnp.mean(losses)


def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    batch_size=64,
    num_test_samps: int = 1_000,
    num_train_steps=1_000_000,
    sweep: int = 10_000,
    sample=False,
    resume=True,
) -> None:

    # data
    dataset = Dataset(data_path, num_test_samps)
    encoder = Encoder(dataset)

    all_data_str = dataset.read_all_data()
    all_data_input, all_data_label = encoder.encode(all_data_str)

    test_input = all_data_input[dataset.test_idx]
    test_label = all_data_label[dataset.test_idx]

    train_idx = list(set(range(len(all_data_label))) - set(dataset.test_idx))
    train_idx.sort()

    train_input = all_data_input[train_idx]
    train_label = all_data_label[train_idx]

    # setup model
    model, keys, meta = init_model(
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    _, params_key, dropout_key = keys
    vocab_size, _ = meta

    if (save_path / "params.pkl").exists() and resume:
        params = load_model(save_path)
        last_step = int(get_last_csv_row(save_path / "train_losses.csv")[0])
        logger.info(f"Training resume from {last_step:,}")
        steps = range(last_step, num_train_steps, sweep)
    else:
        blank_idx = get_sample_idx(batch_size, len(train_idx))
        blank_batch_input = train_input[blank_idx]

        key_data = {"params": params_key}
        blank_model = model.init(key_data, blank_batch_input, training=True)
        params = blank_model["params"]

        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        write_stat(save_path / "stats.txt", "n_params", f"{n_params:,}")
        logger.info(f"Model initialized. Parameter count: {n_params}")

        steps = range(0, num_train_steps, sweep)

    state = init_train_state(model, params, dropout_key)

    # training
    logger.info("\n--- Starting Training ---")
    for step in steps:
        print(step)
        inputs_10k = []
        labels_10k = []

        for _ in range(sweep):
            sample_idx = get_sample_idx(batch_size, len(train_idx))
            inputs_10k.append(train_input[sample_idx])
            labels_10k.append(train_label[sample_idx])

        jnp_inputs = jnp.stack(inputs_10k)
        jnp_labels = jnp.stack(labels_10k)

        # Run 1,000 steps entirely on the GPU in one shot
        jax.profiler.start_trace("/enna/nobackup/danielb/data/results/jax-trace")
        state, loss = train_10k_steps(state, jnp_inputs, jnp_labels)
        loss.block_until_ready()
        jax.profiler.stop_trace()
        print("[INFO] Profiler trace saved! Exiting script to check TensorBoard.")
        exit()
        print("done 10,000 ;)")
        # sample_idx = get_sample_idx(batch_size, len(train_idx))
        # batch_input = train_input[sample_idx]
        # batch_label = train_label[sample_idx]

        # state, loss = train_step_auto_regression(state, batch_input, batch_label)

        if (step + sweep) % sweep == 0 or (step + sweep) == num_train_steps:
            msg = f"Step {step + sweep:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
            logger.info(msg)

            test_loss = get_test_loss(
                state,
                test_input,
                test_label,
                vocab_size,
            )

            write_loss(save_path / "train_losses.csv", step + sweep, float(loss))
            write_loss(save_path / "test_losses.csv", step + sweep, float(test_loss))
            save_model(save_path, state)

        if sample and (step + sweep) % 100_000 == 0:
            sample_model(
                data_path,
                save_path,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                samps_to_gen=1_000,
                gen_its=1,
                tag=f"{step+sweep:,}",
            )

    logger.info("\n Training finished.")

    save_model(save_path, state)


# main
def main_train_tet():
    train = True
    sample = True

    logging.basicConfig(level=logging.INFO)
    send_ntfy(
        "usyd-knottedness",
        "Training Started",
        f"Started training for SGD models on dehydration data",
    )

    Ns = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]
    for N in Ns:
        logger.info(f"\n\n=== N_TET = {N} ===")
        processed_data_home = data_home / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / f"spheres_{N}.hdf5"

        save_path = data_home

        train_time = 0
        sample_time = 0

        if train:
            save_path = (
                data_home
                / "sgd_models_dehydration"
                / f"spheres_512emb_6block_4head_{N}tet"
            )
            save_path.mkdir(parents=True, exist_ok=True)

            tic = time.time()
            train_model(data_path, save_path)
            toc = time.time()

            train_time = toc - tic
            logger.info(f"Training time: {train_time:.2f} seconds")

        if sample:
            tic = time.time()
            sample_model(data_path, save_path, samps_to_gen=1_000, gen_its=20)
            toc = time.time()

            sample_time = toc - tic
            logger.info(f"Sampling time: {sample_time:.2f} seconds")

        send_ntfy(
            "usyd-knottedness",
            f"Training Finished for N={N}",
            f"Finished training for N={N}. Training time: {train_time:.2f} seconds. Sampling time: {sample_time:.2f} seconds.",
        )

    send_ntfy(
        "usyd-knottedness",
        "All Training Finished",
        f"Finished training for all N.",
    )


def main_train_long():
    logging.basicConfig(level=logging.INFO)
    send_ntfy(
        "usyd-knottedness",
        "Long Training Started",
        f"Started long training for SGD models on dehydration data",
    )

    processed_data_path = data_home / "input_data" / "dehydration" / "processed"
    data_path = processed_data_path / f"spheres_10.hdf5"

    train_time = 0
    sample_time = 0

    save_path = (
        data_home
        / "results"
        / "sgd_models_dehydration"
        / "long_train"
        / "spheres_512emb_6block_4head_10tet"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    # Train
    tic = time.time()
    train_model(data_path, save_path, num_train_steps=10_000_000, sample=True)
    toc = time.time()

    train_time = toc - tic
    logger.info(f"Training time: {train_time:.2f} seconds")

    # Sample
    tic = time.time()
    sample_model(data_path, save_path, samps_to_gen=1_000, gen_its=20, tag="final")
    toc = time.time()

    sample_time = toc - tic
    logger.info(f"Sampling time: {sample_time:.2f} seconds")

    # NTFY
    send_ntfy(
        "usyd-knottedness",
        "Training Finished",
        f"Finished long training.",
    )


def main_train_scale():
    train = True
    sample = True

    embs = {"xs": 256, "s": 384, "m": 512, "l": 768, "xl": 1024}
    blocks = {"xs": 4, "s": 6, "m": 12, "l": 16, "xl": 24}
    heads = {"xs": 4, "s": 6, "m": 8, "l": 12, "xl": 16}
    itts = {
        "xs": 500_000,
        "s": 1_500_000,
        "m": 5_000_000,
        "l": 12_500_000,
        "xl": 12_500_000,
    }

    logging.basicConfig(level=logging.INFO)
    send_ntfy(
        "usyd-knottedness",
        "Scale Training Started",
        f"Started training for SGD models on dehydration data",
    )

    sizes = ["xs"]
    for size in sizes:
        emb = embs[size]
        block = blocks[size]
        head = heads[size]

        processed_data_home = data_home / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / "spheres_10.hdf5"

        save_path = data_home

        train_time = 0
        sample_time = 0

        if train:
            save_path = (
                data_home
                / "results"
                / "sgd_models_dehydration"
                / "archive"
                / "scale"
                / f"spheres_{emb}emb_{block}block_{head}head_10tet"
            )
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directoy: {save_path.resolve()}")
            write_stat(save_path / "stats.txt", "size:", size)

            tic = time.time()

            train_model(
                data_path,
                save_path,
                d_model=emb,
                num_heads=head,
                num_layers=block,
                batch_size=16,
                # num_train_steps=itts[size],
                num_train_steps=200,
                sweep=100,
                sample=True,
                resume=False,
            )
            toc = time.time()

            train_time = toc - tic
            logger.info(f"Training time: {train_time:.2f} seconds")

        if sample:
            tic = time.time()
            sample_model(
                data_path,
                save_path,
                d_model=emb,
                num_heads=head,
                num_layers=block,
                samps_to_gen=1_000,
                gen_its=20,
            )
            toc = time.time()

            sample_time = toc - tic
            logger.info(f"Sampling time: {sample_time:.2f} seconds")

        send_ntfy(
            "usyd-knottedness",
            f"Training Finished for size={size}",
            f"Finished training for size={size}. Training time: {train_time:.2f} seconds. Sampling time: {sample_time:.2f} seconds.",
        )

    send_ntfy(
        "usyd-knottedness",
        "All Scale Training Finished",
        f"Finished training for all scales.",
    )


if __name__ == "__main__":
    if "long" in sys.argv:
        main_train_long()
    if "tet" in sys.argv:
        main_train_tet()
    if "scale" in sys.argv:
        main_train_scale()
