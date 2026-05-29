import logging
import pathlib
import sys
import time
from functools import partial

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
from pachner_traversal.types import ObjType
from pachner_traversal.utils import (
    create_sample_schedule,
    data_root,
    get_sample_idx,
    normalize,
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
    dset_name: ObjType = "edge_degree_variance",
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    use_mask: bool = True,
    output_size: int | None = None,
    batch_size: int = 64,
    epochs: int = 64,
    num_test_samps: int = 1_000,
    num_train_steps: int = 1_000_000,
    sweep: int = 10_000,
    learning_rate: float = 0.0005,
    resume: bool = True,
) -> None:

    # Dataset and Encoder.
    dataset = Dataset(data_path, num_test_samps)
    encoder = Encoder(dataset)

    # Read in signatures and encode.
    all_data_input_str = dataset.read_all_data()
    all_data_input, _ = encoder.encode(all_data_input_str)

    # Read in target value and normalize.
    all_data_target_value_raw = dataset.read_all_data(dset_name=dset_name)
    all_data_target_value = np.array(all_data_target_value_raw)
    all_data_target_value = normalize(all_data_target_value)

    # Split train and test data.
    test_input = all_data_input[dataset.test_idx]
    test_target_value = all_data_target_value[dataset.test_idx]

    train_idx = list(set(range(len(all_data_target_value))) - set(dataset.test_idx))
    train_idx.sort()

    train_input = all_data_input[train_idx]
    train_target_value = all_data_target_value[train_idx]

    # Setup model.
    model, keys, _ = init_model(
        ScalarTransformer,
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        use_mask=use_mask,
        output_size=output_size,
    )
    _, params_key, dropout_key = keys

    # Resume / init parameters.
    resumed, meta, steps, params = init_params(
        model,
        params_key,
        save_path,
        train_input,
        batch_size,
        num_train_steps,
        sweep,
        resume,
    )

    state = init_train_state(
        model,
        params,
        dropout_key,
        train_steps=num_train_steps,
        peak_learning_rate=learning_rate,
    )

    if resumed:
        logger.info(f"Training resume from {meta:,}")
    else:
        write_stat(save_path / "stats.txt", "n_params", f"{meta:,}")
        logger.info(f"Model initialized. Parameter count: {meta}")

    schedule = create_sample_schedule(
        batch_size,
        dataset_size=len(train_input),
        epochs=epochs,
        num_itts=num_train_steps,
    )

    # Training.
    logger.info("\n--- Starting Training ---")
    for step in steps:
        # Collect data.
        inputs_sweep = []
        target_values_sweep = []

        for i in range(sweep):
            sample_idx = get_sample_idx(schedule, batch_size, step + i)
            inputs_sweep.append(train_input[sample_idx])
            target_values_sweep.append(train_target_value[sample_idx])

        try:
            jnp_inputs = jnp.stack(inputs_sweep)
            jnp_target_values = jnp.stack(target_values_sweep)
        except Exception as _:
            m = "Error stacking inputs or target values. Likely incomplete sweep size."
            logger.error(m)
            continue

        # Run training steps.
        state, losses = train_sweep_steps(
            train_step_scalar_regression,
            state,
            jnp_inputs,
            jnp_target_values,
        )
        loss = jnp.mean(losses)

        # Log progress.
        msg = f"Step {step + sweep:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
        logger.info(msg)

        # Get test loss.
        test_loss = get_test_loss(
            state,
            test_input,
            test_target_value,
        )

        # Write data.
        write_loss(
            save_path / "train_losses.csv",
            step + sweep,
            float(loss),
        )
        write_loss(
            save_path / "test_losses.csv",
            step + sweep,
            float(test_loss),
        )
        save_model(save_path, state)

        del loss
        del losses
        del test_loss


# Main functions.
def main_train(lr):
    N = 10

    obj_funcs: list[ObjType] = [
        "count_5_deg",
        "count_4_deg",
        "count_3_deg",
        "count_2_deg",
        "count_1_deg",
        "edge_degree_variance",
        "loop_count",
        "det_alexander",
    ]

    logging.basicConfig(level=logging.INFO)

    for obj_func in obj_funcs:
        logger.info(f"\n\n--- OBJ: `{obj_func}` ---")
        processed_data_home = data_root / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / f"spheres_{N}.hdf5"

        save_path = (
            data_root
            / "results"
            / "sgd_models_dehydration"
            / "scalar_regression"
            / obj_func
            / f"{lr}lr_48epoch_512batch"
            / f"spheres_512emb_6block_4head_{N}tet"
        )

        tic = time.time()
        train_model(
            data_path,
            save_path,
            dset_name=obj_func,
            d_model=512,
            num_layers=6,
            num_heads=4,
            use_mask=True,
            output_size=64,
            batch_size=512,
            epochs=48,
            num_test_samps=5_000,
            num_train_steps=93_264,
            sweep=300,
            learning_rate=lr,
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
    if "xhi" in sys.argv:
        main_train(1e-3)
    if "high" in sys.argv:
        main_train(3e-4)
    if "med" in sys.argv:
        main_train(1e-4)
    if "low" in sys.argv:
        main_train(3e-5)
    if "xlo" in sys.argv:
        main_train(1e-5)
