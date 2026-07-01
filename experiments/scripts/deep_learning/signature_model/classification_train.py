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
    train_step_classification,
    train_sweep_steps,
)
from pachner_traversal.utils import (
    create_sample_schedule,
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
    test_loss = optax.sigmoid_binary_cross_entropy(logits, test_batch_label).mean()
    return test_loss


# critical functions
def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    use_mask: bool = True,
    output_size: int | None = None,
    batch_size: int = 64,
    epochs: int = 64,
    num_train_steps: int = 1_000_000,
    sweep: int = 10_000,
    learning_rate: float = 0.0005,
    resume: bool = True,
) -> None:

    # Dataset and Encoder.
    dataset = Dataset(data_path, 1)
    encoder = Encoder(dataset)

    # Read in signatures and encode.
    all_data_input_str = dataset.read_all_data()
    all_data_input, _ = encoder.encode(all_data_input_str)

    # Read in target value.
    all_data_target_value_raw = dataset.read_all_data(dset_name="is_manifold")
    all_data_target_value = np.array(all_data_target_value_raw)

    # Read in triangulation type, test status.
    all_data_tri_type_raw = dataset.read_all_data(dset_name="triangulation_type")
    all_data_tri_type = np.array(all_data_tri_type_raw)
    unique_tri_types = np.unique(all_data_tri_type)
    all_data_is_test_raw = dataset.read_all_data(dset_name="is_test")
    all_data_is_test = np.array(all_data_is_test_raw)

    # Split train and test data.
    test_inputs = {}
    test_target_values = {}
    for tri_type in unique_tri_types:
        filt = (all_data_is_test == 1) & (all_data_tri_type == tri_type)
        test_input = all_data_input[filt]
        test_target_value = all_data_target_value[filt]
        test_inputs[tri_type] = test_input
        test_target_values[tri_type] = test_target_value

    train_input = all_data_input[all_data_is_test == 0]
    train_target_value = all_data_target_value[all_data_is_test == 0]

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
        dataset,
        encoder,
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
            train_step_classification,
            state,
            jnp_inputs,
            jnp_target_values,
        )
        loss = jnp.mean(losses)

        # Log progress.
        msg = f"Step {step + sweep:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
        logger.info(msg)

        # Get test loss.
        for tri_type in unique_tri_types:
            test_input = test_inputs[tri_type]
            test_target_value = test_target_values[tri_type]
            test_loss = get_test_loss(
                state,
                test_input,
                test_target_value,
            )
            write_loss(
                save_path / f"test_losses_{tri_type}.csv",
                step + sweep,
                float(test_loss),
            )
            del test_loss

        # Write data.
        write_loss(
            save_path / "train_losses.csv",
            step + sweep,
            float(loss),
        )

        save_model(save_path, state)

        del loss
        del losses


# Main functions.
def main_train(lr):
    N = 10

    logging.basicConfig(level=logging.INFO)

    processed_data_home = data_root / "input_data" / "dehydration" / "processed"
    data_path = processed_data_home / f"classification_{N}.hdf5"

    save_path = (
        data_root
        / "results"
        / "sgd_models_dehydration"
        / "classification"
        / f"{lr}lr_8epoch_512batch"
        / f"512emb_6block_4head_{N}tet"
    )

    tic = time.time()
    train_model(
        data_path,
        save_path,
        d_model=512,
        num_layers=6,
        num_heads=4,
        use_mask=True,
        output_size=64,
        batch_size=512,
        epochs=8,
        num_train_steps=250,
        sweep=250,
        learning_rate=lr,
        resume=False,
    )
    toc = time.time()

    train_time = toc - tic
    logger.info(f"Training time: {train_time:.2f} seconds")

    message = f"Training time: {train_time:.2f} seconds."
    send_ntfy(
        "usyd-knottedness",
        "Finished training for classification.",
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
