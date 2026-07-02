import logging
import pathlib
import sys
import time
from dataclasses import asdict, dataclass

import jax.numpy as jnp
import numpy as np
import optax
from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import ScalarTransformer
from pachner_traversal.transformer_training import (
    BaseConfig,
    create_get_test_loss,
    create_train_step,
    init_model,
    init_params,
    init_train_state,
    train_sweep_steps,
)
from pachner_traversal.types import ObjType
from pachner_traversal.utils import (
    create_sample_schedule,
    get_data_root,
    get_sample_idx,
    logger_config,
    normalize,
    read_config,
    save_model,
    send_ntfy,
    silence_jax,
    write_loss,
    write_stat,
)


@dataclass
class ScalarRegressionConfig(BaseConfig):
    dset_name: ObjType = "edge_degree_variance"
    output_size: int = 64
    use_mask: bool = True


logger = logging.getLogger(__name__)
loss_metric_fn = optax.squared_error

get_test_loss = create_get_test_loss(loss_metric_fn)
train_step = create_train_step(loss_metric_fn)


# helper functions
def load_data(data_path, num_test_samps, dset_name):
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

    assert dataset.test_idx is not None, "No test idx specified."
    train_idx = list(set(range(len(all_data_target_value))) - set(dataset.test_idx))
    train_idx.sort()

    train_input = all_data_input[train_idx]
    train_target_value = all_data_target_value[train_idx]

    return (
        dataset,
        encoder,
        train_input,
        train_target_value,
        test_input,
        test_target_value,
    )


# critical functions
def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    dset_name: ObjType,
    d_model: int,
    num_layers: int,
    num_heads: int,
    use_mask: bool,
    output_size: int | None,
    use_mup: bool,
    base_d_model: int,
    batch_size: int,
    epochs: int,
    num_train_steps: int,
    learning_rate: float,
    sweep: int,
    num_test_samps: int,
    intrem_train_loss: bool,
    intrem_test_loss: bool,
    final_test_loss: bool,
    final_save_model: bool,
    **kwargs,
) -> tuple[float | None, int]:
    # Load data.
    dataset, encoder, train_input, train_target_value, test_input, test_target_value = (
        load_data(data_path, num_test_samps, dset_name)
    )

    # Initialise model.
    model, keys, _ = init_model(
        ScalarTransformer,
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        use_mask=use_mask,
        output_size=output_size,
        use_mup=use_mup,
        base_d_model=base_d_model,
    )
    _, params_key, dropout_key = keys

    # Initialise parameters.
    model_size, steps, params = init_params(
        model,
        params_key,
        save_path,
        dataset,
        encoder,
        batch_size,
        num_train_steps,
        sweep,
    )

    # Initialise train state.
    state = init_train_state(
        model,
        params,
        dropout_key,
        train_steps=num_train_steps,
        peak_learning_rate=learning_rate,
        use_mup=use_mup,
        base_d_model=base_d_model,
    )

    write_stat(save_path / "stats.txt", "n_params", f"{model_size:,}")
    logger.info(f"Model initialized. Parameter count: {model_size:,}")

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
            if len(sample_idx) == batch_size:
                inputs_sweep.append(train_input[sample_idx])
                target_values_sweep.append(train_target_value[sample_idx])
            else:
                awk_size = len(sample_idx)
                logger.warning(f"{awk_size} akward samples found, discarding.")
                break

        actual_sweep = len(inputs_sweep)
        jnp_inputs = jnp.stack(inputs_sweep)
        jnp_target_values = jnp.stack(target_values_sweep)

        # Run training steps.
        state, losses = train_sweep_steps(
            train_step,
            state,
            jnp_inputs,
            jnp_target_values,
        )
        loss = jnp.mean(losses)

        # Log progress.
        snum = step + actual_sweep
        msg = f"Step {snum:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
        logger.info(msg)

        # Intrem results.
        if intrem_train_loss:
            write_loss(
                save_path / "train_losses.csv",
                step + actual_sweep,
                float(loss),
            )
        del loss
        del losses

        if intrem_test_loss:
            test_loss = get_test_loss(
                state,
                test_input,
                test_target_value,
            )
            write_loss(
                save_path / "test_losses.csv",
                step + actual_sweep,
                float(test_loss),
            )
            del test_loss

    if final_test_loss:
        test_loss = get_test_loss(
            state,
            test_input,
            test_target_value,
        )
        test_loss_float = float(test_loss)
        del test_loss
    else:
        test_loss_float = None

    if final_save_model:
        save_model(save_path, state)

    return test_loss_float, model_size


# Main functions.
def main_train(config_path: pathlib.Path, run_model_tag: str, nci: bool = False):
    logging.basicConfig(**logger_config)
    silence_jax()

    config_data = read_config(config_path)
    data_root = get_data_root(nci)
    config_data["data_path"] = data_root / config_data["data_path_stem"]
    config_data["save_path"] = data_root / config_data["save_path_stem"]
    config_data["nci"] = nci
    if (
        config_data["run_model_tag"] != run_model_tag
        or config_data["run_model_tag"] == "ignore"
    ):
        return

    config = ScalarRegressionConfig.from_dict(config_data)

    tic = time.time()
    train_model(**asdict(config))
    toc = time.time()

    train_time = toc - tic
    logger.info(f"Training time: {train_time:.2f} seconds")

    if not nci:
        message = f"Training time: {train_time:.2f} seconds."
        send_ntfy(
            "usyd-knottedness",
            f"Finished training for {config.dname}.",
            message,
        )


if __name__ == "__main__":
    nci = False
    data_root = get_data_root(nci)
    config_path = data_root.parent / "experiments" / "configs" / "scalar_regression"
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    for config_file in config_path.glob("*.yaml"):
        main_train(config_file, tag, nci=nci)
