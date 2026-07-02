"""
Specialised version of auto regression train to specifically train for 15 tetrahedra.
The only difference with this file is that some of the data for the dataset is precomputed
to reduce loading time.
"""

import logging
import pathlib
import sys
import time

import jax.numpy as jnp
import numpy as np
from pachner_traversal.data import char_list15
from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import (
    Transformer,
)
from pachner_traversal.transformer_training import (
    init_model,
    init_params,
    init_train_state,
    train_step_auto_regression,
    train_sweep_steps,
)
from pachner_traversal.utils import (
    get_data_root,
    logger_config,
    silence_jax,
    write_loss,
    write_stat,
)

from .auto_regression_train import get_test_loss

logger = logging.getLogger(__name__)


# critical functions
def setup_model(
    data_path: pathlib.Path,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    num_test_samps: int = 10_000,
):

    # data
    logger.debug("Setting up dataset")
    dataset = Dataset(
        data_path,
        num_test_samps,
        data_size=160_036_916,
        chars=char_list15,
        max_len=41,
        store_in_memory=True,
    )
    logger.debug("Setting up encoder")
    encoder = Encoder(dataset)

    logger.debug("Setting up train idx")
    mask = np.ones(len(dataset), dtype=bool)
    mask[dataset.test_idx] = False

    train_idx = np.arange(len(dataset))[mask]

    np.random.seed(42)
    np.random.shuffle(train_idx)
    train_idx = train_idx.tolist()

    logger.debug("Loading limited test data")
    test_samples = dataset.test_data
    logger.debug("Encoding limited test data")
    test_input, test_label = encoder.encode(test_samples)

    # setup model
    logger.debug("Initialising model")
    model, keys, meta = init_model(
        Transformer,
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    _, params_key, dropout_key = keys
    vocab_size, seq_len = meta

    return (
        dataset,
        encoder,
        model,
        params_key,
        dropout_key,
        train_idx,
        test_input,
        test_label,
        vocab_size,
        seq_len,
    )


def setup_batch(
    dataset,
    encoder,
    train_idx,
    sweep,
    batch_size,
    step,
):
    inputs_sweep = []
    labels_sweep = []
    sample_idx_sweep = []

    for i in range(sweep):
        start_pos = (step + i) * batch_size
        end_pos = start_pos + batch_size

        sample_idx = list(range(start_pos, end_pos))
        sample_idx_sweep.append(sample_idx)

    sample_idx_sweep_flat = np.array(sample_idx_sweep).flatten()

    logger.debug("Setting up read_idx")
    read_idx = train_idx[sample_idx_sweep_flat]
    logger.debug(f"Reading {len(sample_idx_sweep_flat):,} lines")
    sweep_samples = dataset.read_lines(read_idx)

    logger.debug("Encoding")
    sweep_samples = np.array(sweep_samples).reshape(-1, batch_size)
    for i in range(sweep):
        b_input, b_label = encoder.encode(sweep_samples[i])
        inputs_sweep.append(b_input)
        labels_sweep.append(b_label)

    jnp_inputs = jnp.stack(inputs_sweep)
    jnp_labels = jnp.stack(labels_sweep)

    return jnp_inputs, jnp_labels


def train_model(
    save_path: pathlib.Path,
    batch_size=512,
    num_train_steps=30_000,
    sweep: int = 300,
    learning_rate: float = 1e-4,
    model_setup: tuple = (),
    d_model: int = 64,
) -> tuple[float, int] | None:
    dataset = model_setup[0]
    encoder = model_setup[1]
    model = model_setup[2]
    params_key = model_setup[3]
    dropout_key = model_setup[4]
    train_idx = np.array(model_setup[5])
    test_input = model_setup[6]
    test_label = model_setup[7]
    vocab_size = model_setup[8]

    bulk_num_train_steps = sweep * (num_train_steps // sweep)

    logger.debug("Initialising parameters")
    meta, steps, params = init_params(
        model,
        params_key,
        save_path,
        dataset,
        encoder,
        batch_size,
        bulk_num_train_steps,
        sweep,
    )

    logger.debug("Initialising train state")
    state = init_train_state(
        model,
        params,
        dropout_key,
        train_steps=num_train_steps,
        peak_learning_rate=learning_rate,
        d_model=d_model,
    )

    write_stat(save_path / "stats.txt", "n_params", f"{meta:,}")
    logger.info(f"Model initialized. Parameter count: {meta:,}")

    # Training.
    logger.info("\n--- Starting Training ---")
    logger.debug(f"`steps`: {steps}")
    for step in steps:
        # Setup batch.
        try:
            jnp_inputs, jnp_labels = setup_batch(
                dataset,
                encoder,
                train_idx,
                sweep,
                batch_size,
                step,
            )
        except Exception as e:
            logger.error(f"Error stacking inputs/labels at step {step}: {e}")
            continue

        # Train sweep.
        logger.debug("Training...")
        state, losses = train_sweep_steps(
            train_step_auto_regression,
            state,
            jnp_inputs,
            jnp_labels,
        )
        loss = jnp.mean(losses)

        msg = f"Step {step + sweep:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
        logger.info(msg)

        write_loss(
            save_path / "train_losses.csv",
            step + sweep,
            float(loss),
        )

        del loss
        del losses

    if num_train_steps > bulk_num_train_steps:
        try:
            jnp_inputs, jnp_labels = setup_batch(
                dataset,
                encoder,
                train_idx,
                num_train_steps - bulk_num_train_steps,
                batch_size,
                bulk_num_train_steps,
            )
        except Exception as e:
            logger.error(f"Error stacking inputs/labels at final step: {e}")
            return

        # Train sweep.
        state, losses = train_sweep_steps(
            train_step_auto_regression,
            state,
            jnp_inputs,
            jnp_labels,
        )
        loss = jnp.mean(losses)

        msg = f"Step {num_train_steps:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"

    test_loss = get_test_loss(
        state,
        test_input,
        test_label,
        vocab_size,
    )
    test_loss_float = float(test_loss)
    del test_loss

    write_loss(
        save_path / "test_losses.csv",
        num_train_steps,
        test_loss_float,
    )

    # sample_model(
    #     encoder,
    #     state,
    #     seq_len,
    #     save_path,
    #     gen_its=16,
    #     samps_to_gen=1_000,
    # )

    logger.info("\n Training finished.")

    return test_loss_float, meta


def main_train_scale(models):
    logging.basicConfig(**logger_config)
    silence_jax()

    data_root = get_data_root()

    for model in models:
        processed_data_home = data_root / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / "spheres_15_170m.hdf5"

        lr = 2.5e-3

        emb = model[0]
        block = model[1]
        flops = model[2]

        head = emb // 64
        n_params = 12 * block * emb**2
        toks = flops / (6 * n_params)
        samps = toks / 41
        itts = int(samps / 512)

        if samps >= 150_000_000:
            msg = f"Skipping model {model} as it requires {samps:,.0f} samples."
            logger.warning(msg)
            continue
        if block >= 9:
            msg = f"Skipping model {model} as it is too big."
            logger.warning(msg)
            continue

        logger.info(f"Training model {model} for {itts:,} iterations.")

        model_setup = setup_model(
            data_path=data_path,
            d_model=emb,
            num_layers=block,
            num_heads=head,
            num_test_samps=16_000,
        )

        logger.info(f"number of iterations: {itts:,}")

        global_save_path = (
            data_root / "results" / "sgd_models_dehydration" / "isoflop_scale"
        )

        save_path = (
            global_save_path
            / f"{flops}"
            / f"spheres_{emb}emb_{block}block_{head}head_15tet"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directoy: {save_path.resolve()}")

        # Train model.
        tic = time.time()
        res = train_model(
            save_path,
            batch_size=512,
            num_train_steps=itts,
            sweep=5_000,
            learning_rate=lr,
            model_setup=model_setup,
            d_model=emb,
        )
        toc = time.time()

        if not (global_save_path / "all_res.csv").exists():
            with open(global_save_path / "all_res.csv", "a") as f:
                f.write("emb, block, lr, n_params, test_loss, flops\n")

        if res is not None:
            test_loss_float, meta = res
            with open(global_save_path / "all_res.csv", "a") as f:
                f.write(f"{emb}, {block}, {lr}, {meta}, {test_loss_float}, {flops}\n")

        train_time = toc - tic
        logger.info(f"Training time: {train_time:.2f} seconds")


if __name__ == "__main__":
    all_runs_aspect = [
        (512, 2, 10e15),
        (416, 3, 10e15),
        (352, 4, 10e15),
        (320, 5, 10e15),
        (288, 6, 10e15),
        (256, 8, 10e15),
        (224, 10, 10e15),
    ]

    all_runs = [
        (128, 2, 6e15),
        (192, 3, 6e15),
        (256, 4, 6e15),
        (320, 5, 6e15),
        (384, 6, 6e15),
        (128, 2, 10e15),
        (192, 3, 10e15),
        (256, 4, 10e15),
        (320, 5, 10e15),
        (384, 6, 10e15),
        (192, 3, 30e15),
        (256, 4, 30e15),
        (320, 5, 30e15),
        (384, 6, 30e15),
        (448, 7, 30e15),
        (192, 3, 60e15),
        (256, 4, 60e15),
        (320, 5, 60e15),
        (384, 6, 60e15),
        (448, 7, 60e15),
        (512, 8, 60e15),
        (256, 4, 100e15),
        (320, 5, 100e15),
        (384, 6, 100e15),
        (448, 7, 100e15),
        (512, 8, 100e15),
        (576, 9, 100e15),
        (256, 4, 300e15),
        (320, 5, 300e15),
        (384, 6, 300e15),
        (448, 7, 300e15),
        (512, 8, 300e15),
        (576, 9, 300e15),
        (640, 10, 300e15),
        (704, 11, 300e15),
        (320, 5, 600e15),
        (384, 6, 600e15),
        (448, 7, 600e15),
        (512, 8, 600e15),
        (576, 9, 600e15),
        (640, 10, 600e15),
        (704, 11, 600e15),
        (768, 12, 600e15),
        (320, 5, 1000e15),
        (384, 6, 1000e15),
        (448, 7, 1000e15),
        (512, 8, 1000e15),
        (576, 9, 1000e15),
        (640, 10, 1000e15),
        (704, 11, 1000e15),
        (768, 12, 1000e15),
        (832, 13, 1000e15),
        (384, 6, 3000e15),
        (448, 7, 3000e15),
        (512, 8, 3000e15),
        (576, 9, 3000e15),
        (640, 10, 3000e15),
        (704, 11, 3000e15),
        (768, 12, 3000e15),
        (832, 13, 3000e15),
        (896, 14, 3000e15),
        (960, 15, 3000e15),
        (1024, 16, 3000e15),
    ]

    if "scale_low" in sys.argv:
        main_train_scale(all_runs[26:])
    # if "scale_med" in sys.argv:
    #     main_train_scale()
    # if "scale_high" in sys.argv:
    #     main_train_scale()
