import logging
import pathlib
import sys
import time
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import (
    MinimalTrainState,
    ScalarTransformer,
    init_model,
    init_params,
    init_train_state,
    train_step_scalar_regression,
    train_sweep_steps,
)
from pachner_traversal.utils import (
    data_root,
    get_sample_idx,
    save_model,
    send_ntfy,
    write_loss,
    write_stat,
)

logger = logging.getLogger(__name__)


# jax utility
@partial(jax.jit)
def get_test_loss(
    state: MinimalTrainState,
    test_batch_input: jax.Array,
    test_batch_label: jax.Array,
) -> jax.Array:
    logits = state.apply_fn(
        {"params": state.params},
        test_batch_input,
        training=False,
    )
    test_loss = optax.squared_error(logits, test_batch_label).mean()
    return test_loss


# critical functions
def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    dset_name: Literal[
        "edge_degree_variance", "det_alexander"
    ] = "edge_degree_variance",
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    batch_size=64,
    num_test_samps: int = 1_000,
    num_train_steps=1_000_000,
    sweep=10_000,
    resume=True,
) -> None:

    # data
    dataset = Dataset(data_path, num_test_samps)
    encoder = Encoder(dataset)

    all_data_str = dataset.read_all_data()
    all_data_label_raw = dataset.read_all_data(dset_name=dset_name)
    all_data_label = np.array(all_data_label_raw)
    all_data_input, _ = encoder.encode(all_data_str)

    test_input = all_data_input[dataset.test_idx]
    test_label = all_data_label[dataset.test_idx]

    train_idx = list(set(range(len(all_data_label))) - set(dataset.test_idx))
    train_idx.sort()

    train_input = all_data_input[train_idx]
    train_label = all_data_label[train_idx]

    # setup model
    model, keys, _ = init_model(
        ScalarTransformer,
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    _, params_key, dropout_key = keys

    # resume / init parameters
    resumed, meta, steps, params = init_params(
        save_path,
        resume,
        num_train_steps,
        sweep,
        batch_size,
        train_idx,
        train_input,
        params_key,
        model,
    )

    state = init_train_state(model, params, dropout_key)

    if resumed:
        logger.info(f"Training resume from {meta:,}")
    else:
        write_stat(save_path / "stats.txt", "n_params", f"{meta:,}")
        logger.info(f"Model initialized. Parameter count: {meta}")

    # training
    logger.info("\n--- Starting Training ---")
    for step in steps:
        inputs_sweep = []
        labels_sweep = []

        for _ in range(sweep):
            sample_idx = get_sample_idx(batch_size, len(train_idx))
            inputs_sweep.append(train_input[sample_idx])
            labels_sweep.append(train_label[sample_idx])

        jnp_inputs = jnp.stack(inputs_sweep)
        jnp_labels = jnp.stack(labels_sweep)

        state, loss = train_sweep_steps(
            train_step_scalar_regression,
            state,
            jnp_inputs,
            jnp_labels,
        )

        if (step + sweep) % sweep == 0 or (step + sweep) == num_train_steps:
            msg = f"Step {step + sweep:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
            logger.info(msg)

            test_loss = get_test_loss(
                state,
                test_input,
                test_label,
            )

            write_loss(save_path / "train_losses.csv", step + sweep, float(loss))
            write_loss(save_path / "test_losses.csv", step + sweep, float(test_loss))
            save_model(save_path, state)

    logger.info("\n Training finished.")

    save_model(save_path, state)


# main
def main_train_simple():
    N = 10
    obj_funcs: list[Literal["edge_degree_variance", "det_alexander"]] = [
        "edge_degree_variance",
        "det_alexander",
    ]

    logging.basicConfig(level=logging.INFO)

    for obj_func in obj_funcs:
        logger.info(f"\n\n=== OBJ = {obj_func} ===")
        processed_data_home = data_root / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / f"spheres_{N}.hdf5"

        save_path = (
            data_root
            / "results"
            / "sgd_models_dehydration"
            / "scalar_simple"
            / obj_func
            / f"spheres_512emb_6block_4head_{N}tet"
        )
        save_path.mkdir(parents=True, exist_ok=True)

        tic = time.time()
        train_model(
            data_path,
            save_path,
            dset_name=obj_func,
            d_model=512,
            num_layers=6,
            num_heads=4,
            batch_size=16,
            num_test_samps=5_000,
            num_train_steps=10_000_000,
            resume=False,
        )
        toc = time.time()

        train_time = toc - tic
        logger.info(f"Training time: {train_time:.2f} seconds")

        message = f"Training time: {train_time:.2f} seconds."
        send_ntfy(
            "usyd-knottedness",
            f"Finished training for {obj_func}.",
            message,
        )


if __name__ == "__main__":
    if "simple" in sys.argv:
        main_train_simple()
