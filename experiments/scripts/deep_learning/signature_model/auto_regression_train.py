import logging
import pathlib
import shutil
import sys
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from pachner_traversal.data_io_dehydration import Dataset, Encoder
from pachner_traversal.transformer import Transformer
from pachner_traversal.transformer_training import (
    BaseConfig,
    create_get_test_loss,
    create_train_step,
    generate_samples,
    init_model,
    init_params,
    init_train_state,
    train_sweep_steps,
)
from pachner_traversal.utils import (
    create_sample_schedule,
    get_data_root,
    get_sample_idx,
    load_model,
    logger_config,
    read_config,
    save_model,
    send_ntfy,
    silence_jax,
    write_loss,
    write_stat,
)


@dataclass
class AutoRegressionConfig(BaseConfig):
    intrem_sample_size: int | None = None
    final_sample_size: int | None = 1_000


logger = logging.getLogger(__name__)
loss_metric_fn = optax.softmax_cross_entropy_with_integer_labels

get_test_loss = create_get_test_loss(loss_metric_fn)
train_step = create_train_step(loss_metric_fn)


# sample functions
def sample_model(
    encoder: Encoder,
    state,
    seq_len,
    save_path: pathlib.Path,
    gen_its: int = 10,
    samps_to_gen: int = 1_000,
    tag: str | None = None,
) -> None:
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


def sample_model_from_save(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    num_test_samps: int = 1_000,
    gen_its: int = 10,
    samps_to_gen: int = 1_000,
    tag: str | None = None,
    params_fname: str = "params.pkl",
) -> None:
    # setup model
    dataset = Dataset(
        data_path,
        num_test_samps,
        store_in_memory=True,
    )
    encoder = Encoder(dataset)

    params = load_model(save_path, params_fname)

    model, keys, meta = init_model(
        Transformer,
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
    sample_model(
        encoder=encoder,
        state=state,
        seq_len=seq_len,
        save_path=save_path,
        gen_its=gen_its,
        samps_to_gen=samps_to_gen,
        tag=tag,
    )


# Critical functions.
def load_data(data_path, num_test_samps):
    # Dataset and Encoder.
    dataset = Dataset(data_path, num_test_samps, store_in_memory=True)
    encoder = Encoder(dataset)

    # Read in signatures and Encode.
    all_data_input_str = dataset.read_all_data()
    all_data_input, all_data_label = encoder.encode(all_data_input_str)

    # Split train and test data.
    test_input = all_data_input[dataset.test_idx]
    test_label = all_data_label[dataset.test_idx]

    assert dataset.test_idx is not None, "No test idx specified."
    train_idx = list(set(range(len(all_data_label))) - set(dataset.test_idx))
    train_idx.sort()

    train_input = all_data_input[train_idx]
    train_label = all_data_label[train_idx]

    return dataset, encoder, train_input, train_label, test_input, test_label


def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    d_model: int,
    num_layers: int,
    num_heads: int,
    use_mup: bool,
    base_d_model: int,
    batch_size: int,
    epochs: int,
    num_train_steps: int | None,
    flops: int | None,
    learning_rate: float,
    sweep: int,
    num_test_samps: int,
    intrem_sample_size: int | None,
    final_sample_size: int | None,
    intrem_train_loss: bool,
    intrem_test_loss: bool,
    final_test_loss: bool,
    final_save_model: bool,
    **kwargs,
) -> tuple[float | None, int] | None:
    # Load data.
    logger.info("Loading data.")
    dataset, encoder, train_input, train_label, test_input, test_label = load_data(
        data_path, num_test_samps
    )

    # Initialise model.
    logger.debug("Initialising model")
    model, keys, meta = init_model(
        Transformer,
        dataset,
        encoder,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        use_mup=use_mup,
        base_d_model=base_d_model,
    )
    _, params_key, dropout_key = keys
    vocab_size, seq_len = meta

    # Initialise parameters.
    logger.debug("Initialising parameters")
    model_size, steps, params, num_train_steps = init_params(
        model,
        params_key,
        save_path,
        dataset,
        encoder,
        batch_size,
        num_train_steps,
        sweep,
        flops=flops,
        seq_len=seq_len,
    )

    # Initialise train state.
    logger.debug("Initialising train state")
    state = init_train_state(
        model,
        params,
        dropout_key,
        train_steps=num_train_steps,
        peak_learning_rate=learning_rate,
        d_model=d_model,
        base_d_model=base_d_model,
        use_mup=use_mup,
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
    logger.debug(f"`steps`: {steps}")
    for step in steps:
        # Setup batch.
        inputs_sweep = []
        labels_sweep = []

        for i in range(sweep):
            sample_idx = get_sample_idx(schedule, batch_size, step + i)
            if len(sample_idx) == batch_size:
                inputs_sweep.append(train_input[sample_idx])
                labels_sweep.append(train_label[sample_idx])
            else:
                awk_size = len(sample_idx)
                logger.warning(f"{awk_size} akward samples found, discarding.")
                break

        actual_sweep = len(inputs_sweep)
        jnp_inputs = jnp.stack(inputs_sweep)
        jnp_labels = jnp.stack(labels_sweep)

        # Train sweep.
        logger.debug("Training...")
        state, losses = train_sweep_steps(
            train_step,
            state,
            jnp_inputs,
            jnp_labels,
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
                test_label,
                vocab_size,
            )
            write_loss(
                save_path / "test_losses.csv",
                num_train_steps,
                float(test_loss),
            )
            del test_loss

        if intrem_sample_size is not None:
            gen_its = intrem_sample_size // 1_000
            sample_model(
                encoder,
                state,
                seq_len,
                save_path,
                gen_its=gen_its,
                samps_to_gen=1_000,
                tag=f"{step + actual_sweep}",
            )

    if final_test_loss:
        test_loss = get_test_loss(
            state,
            test_input,
            test_label,
            vocab_size,
        )
        test_loss_float = float(test_loss)
        del test_loss
    else:
        test_loss_float = None

    if final_sample_size is not None:
        gen_its = final_sample_size // 1_000
        sample_model(
            encoder,
            state,
            seq_len,
            save_path,
            gen_its=gen_its,
            samps_to_gen=1_000,
        )
    if final_save_model:
        save_model(save_path, state)

    logger.info("\n Training finished.")

    return test_loss_float, model_size


def main_train(config_path: pathlib.Path, run_model_tag: str, nci: bool = False):
    logging.basicConfig(**logger_config)
    silence_jax()

    logger.info(f"Considering job {config_path}")

    config_data = read_config(config_path)
    data_root = get_data_root(nci)
    config_data["data_path"] = data_root / config_data["data_path_stem"]
    config_data["save_path"] = data_root / config_data["save_path_stem"]
    config_data["num_train_steps"] = None
    config_data["nci"] = nci
    if (
        config_data["run_model_tag"] != run_model_tag
        or config_data["run_model_tag"] == "ignore"
    ):
        return

    config_data["save_path"].mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, config_data["save_path"] / config_path.name)

    if "base_d_model" not in config_data:
        config_data["base_d_model"] = config_data["d_model"]
    if "num_heads" not in config_data:
        config_data["num_heads"] = config_data["d_model"] // config_data["head_size"]

    logger.debug("Setting up config object")
    config = AutoRegressionConfig.from_dict(config_data)

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
    config_path = data_root.parent / "experiments" / "configs" / "isoflop_aspect_ratio"
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    for config_file in config_path.rglob("*.yaml"):
        main_train(config_file, tag, nci=nci)
