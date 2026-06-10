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
    Transformer,
    generate_samples,
    init_model,
    init_params,
    init_train_state,
    train_step_auto_regression,
    train_sweep_steps,
)
from pachner_traversal.utils import (
    create_sample_schedule,
    get_sample_idx,
    load_model,
    save_model,
    send_ntfy,
    write_loss,
    write_stat,
)
from pachner_traversal.utils import data_root as data_root

logger = logging.getLogger(__name__)

char_list = [
    "Ha",
    "Hb",
    "Hc",
    "Hd",
    "He",
    "Hf",
    "Hg",
    "Hh",
    "Hi",
    "Hj",
    "Hk",
    "Hl",
    "Hm",
    "Hn",
    "Ho",
    "Hp",
    "Hq",
    "Hr",
    "Hs",
    "Ht",
    "Hu",
    "Hv",
    "Hw",
    "Hx",
    "Na",
    "Nb",
    "Nc",
    "Nd",
    "Ne",
    "Nf",
    "Ng",
    "Nh",
    "Ni",
    "Nj",
    "Nk",
    "Nl",
    "Nm",
    "Nn",
    "No",
    "Np",
    "Wa",
    "Wb",
    "Wc",
    "Wd",
    "We",
    "Wf",
    "Wg",
    "Wh",
    "Wi",
    "Wj",
    "Wk",
    "Wl",
    "Wm",
    "Wn",
    "Wo",
    "p",
]


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


def train_model(
    data_path: pathlib.Path,
    save_path: pathlib.Path,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 4,
    batch_size=512,
    epochs=16,
    num_test_samps: int = 10_000,
    num_train_steps=30_000,
    sweep: int = 300,
    learning_rate: float = 1e-4,
    samp_freq=10,
    sample=True,
    resume=False,
    low_mem=False,
) -> None:

    # data
    logger.debug("Setting up dataset")
    dataset = Dataset(
        data_path,
        num_test_samps,
        data_size=160_036_916,
        chars=char_list,
        max_len=41,
        store_in_memory=True,
    )
    logger.debug("Setting up encoder")
    encoder = Encoder(dataset)

    train_idx = list(set(range(len(dataset))) - set(dataset.test_idx))
    train_idx.sort()

    if not low_mem:
        logger.debug("Loading all data")
        all_data_str = dataset.read_all_data()
        all_data_input, all_data_label = encoder.encode(all_data_str)

        test_input = all_data_input[dataset.test_idx]
        test_label = all_data_label[dataset.test_idx]

        train_input = all_data_input[train_idx]
        train_label = all_data_label[train_idx]
    else:
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
    vocab_size, _ = meta

    logger.debug("Initialising parameters")
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

    logger.debug("Initialising train state")
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
        logger.info(f"Model initialized. Parameter count: {meta:,}")

    logger.debug("Creating sample schedule")
    schedule = create_sample_schedule(
        batch_size,
        dataset_size=len(train_idx),
        epochs=epochs,
        num_itts=num_train_steps,
    )

    # training
    logger.info("\n--- Starting Training ---")
    sam_counter = 0
    for step in steps:
        inputs_sweep = []
        labels_sweep = []
        sample_idx_sweep = []

        if step <= 2:
            logger.debug("Starting batch gen")

        for i in range(sweep):
            sample_idx = get_sample_idx(schedule, batch_size, step + i)
            sample_idx_sweep.append(sample_idx)

        if low_mem:
            sample_idx_sweep_flat = np.array(sample_idx_sweep).flatten()
            logger.debug(f"Reading {len(sample_idx_sweep_flat):,} lines")
            sweep_samples = dataset.read_lines(
                np.array(train_idx)[sample_idx_sweep_flat]
            )

            logger.debug("Encoding")
            sweep_samples = np.array(sweep_samples).reshape(-1, batch_size)
            for i in range(sweep):
                b_input, b_label = encoder.encode(sweep_samples[i])
                inputs_sweep.append(b_input)
                labels_sweep.append(b_label)
        else:
            for sample_idx in sample_idx_sweep:
                inputs_sweep.append(train_input[sample_idx])  # type: ignore
                labels_sweep.append(train_label[sample_idx])  # type: ignore

        try:
            jnp_inputs = jnp.stack(inputs_sweep)
            jnp_labels = jnp.stack(labels_sweep)
        except Exception as e:
            logger.error(f"Error stacking inputs/labels at step {step}: {e}")
            continue

        if step <= 2:
            logger.info("Finished batch gen")

        state, losses = train_sweep_steps(
            train_step_auto_regression,
            state,
            jnp_inputs,
            jnp_labels,
        )
        loss = jnp.mean(losses)

        msg = f"Step {step + sweep:,}/{num_train_steps:,}, Loss: {float(loss):.4f}"
        logger.info(msg)

        logger.debug("Get test loss")
        test_loss = get_test_loss(
            state,
            test_input,
            test_label,
            vocab_size,
        )

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

        if sam_counter % samp_freq == 0 and sample:
            logger.debug("Sample model")
            sample_model(
                data_path,
                save_path,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                samps_to_gen=1_000,
                gen_its=1,
                tag=f"{step + sweep:,}",
            )
        sam_counter += 1

    logger.info("\n Training finished.")

    save_model(save_path, state)


def main_train_scale(lr):
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)

    train = True
    sample = True

    embs = {"xs": 256, "s": 384, "m": 512, "l": 768, "xl": 1024}
    blocks = {"xs": 4, "s": 6, "m": 12, "l": 16, "xl": 24}
    heads = {"xs": 4, "s": 6, "m": 8, "l": 12, "xl": 16}
    itts = {"xs": 10_000, "s": 40_000, "m": 110_000, "l": 300_000, "xl": 300_000}
    samp_freqs = {"xs": 1, "s": 2, "m": 5, "l": 10, "xl": 10}
    sweeps = {"xs": 200, "s": 400, "m": 400, "l": 600, "xl": 600}

    sizes = ["xs"]
    for size in sizes:
        emb = embs[size]
        block = blocks[size]
        head = heads[size]

        processed_data_home = data_root / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / "spheres_15_16m.hdf5"

        save_path = data_root

        train_time = 0
        sample_time = 0

        if train:
            save_path = (
                data_root
                / "results"
                / "sgd_models_dehydration"
                / "archive"
                / "scale"
                / f"{size}"
                / f"{lr}"
                / f"spheres_{emb}emb_{block}block_{head}head_15tet"
            )
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directoy: {save_path.resolve()}")
            write_stat(save_path / "stats.txt", "size:", size)

            tic = time.time()

            train_model(
                data_path,
                save_path,
                d_model=emb,
                num_layers=block,
                num_heads=head,
                batch_size=512,
                epochs=1,
                num_test_samps=10_000,
                num_train_steps=itts[size],
                sweep=sweeps[size],
                samp_freq=samp_freqs[size],
                learning_rate=lr,
                sample=True,
                resume=False,
                low_mem=True,
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

        message = (
            f"Training time: {train_time:.2f} seconds."
            f"Sampling time: {sample_time:.2f} seconds."
        )
        send_ntfy(
            "usyd-knottedness",
            f"Finished training for size={size}.",
            message,
        )


if __name__ == "__main__":
    if "scale_xlo" in sys.argv:
        main_train_scale(1e-4)
    if "scale_low" in sys.argv:
        main_train_scale(3e-4)
    if "scale_med" in sys.argv:
        main_train_scale(1e-3)
    if "scale_high" in sys.argv:
        main_train_scale(3e-3)
