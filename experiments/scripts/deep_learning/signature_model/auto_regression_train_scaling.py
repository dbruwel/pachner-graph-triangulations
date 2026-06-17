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
    init_fine_tune_state,
    init_model,
    init_params,
    init_train_state_scale,
    train_step_auto_regression,
    train_sweep_steps,
)
from pachner_traversal.utils import data_root as data_root
from pachner_traversal.utils import (
    load_model,
    save_model,
    send_ntfy,
    write_loss,
    write_stat,
)

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
        data_size=160_036_916,
        chars=char_list,
        max_len=41,
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

    state = init_train_state_scale(model, params, dropout_key)

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
        chars=char_list,
        max_len=41,
        store_in_memory=True,
    )
    logger.debug("Setting up encoder")
    encoder = Encoder(dataset)

    train_idx = list(set(range(len(dataset))) - set(dataset.test_idx))
    np.random.seed(42)
    np.random.shuffle(train_idx)
    train_idx = list(train_idx)

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
    logger.debug(f"Reading {len(sample_idx_sweep_flat):,} lines")
    sweep_samples = dataset.read_lines(np.array(train_idx)[sample_idx_sweep_flat])

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
    resume=False,
    resume_from: int | None = None,
    model_setup: tuple = (),
    save_at: list = [1, 2, 4, 8, 16, 32],
) -> list:
    dataset = model_setup[0]
    encoder = model_setup[1]
    model = model_setup[2]
    params_key = model_setup[3]
    dropout_key = model_setup[4]
    train_idx = model_setup[5]

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
        resume_from=resume_from,
    )

    logger.debug("Initialising train state")
    state = init_train_state_scale(
        model,
        params,
        dropout_key,
        peak_learning_rate=learning_rate,
        resume=resume,
    )

    if resumed:
        logger.info(f"Training resume from {meta:,}")
    else:
        write_stat(save_path / "stats.txt", "n_params", f"{meta:,}")
        logger.info(f"Model initialized. Parameter count: {meta:,}")

    # Training.
    logger.info("\n--- Starting Training ---")
    sam_counter = 0
    save_itts = []
    logger.debug(f"`steps`: {steps}")
    for step in steps:
        sam_counter += 1

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

        if sam_counter in save_at:
            # Save if needed.
            save_model(save_path, state, f"_{step + sweep:,}")
            save_itts.append(step + sweep)

    logger.info("\n Training finished.")

    return save_itts


def fine_tune_model(
    save_path: pathlib.Path,
    initial_train_itts: int,
    batch_size=512,
    num_fine_tune_steps=10_000,
    learning_rate: float = 1e-4,
    model_setup: tuple = (),
) -> None:
    dataset = model_setup[0]
    encoder = model_setup[1]
    model = model_setup[2]
    params_key = model_setup[3]
    dropout_key = model_setup[4]
    train_idx = model_setup[5]
    test_input = model_setup[6]
    test_label = model_setup[7]
    vocab_size = model_setup[8]
    seq_len = model_setup[9]

    logger.debug("Initialising parameters")
    _, meta, _, params = init_params(
        model,
        params_key,
        save_path,
        dataset,
        encoder,
        batch_size,
        num_fine_tune_steps,
        num_fine_tune_steps,
        resume=True,
        resume_from=initial_train_itts,
    )

    logger.debug("Initialising train state")
    state = init_fine_tune_state(
        model,
        params,
        dropout_key,
        peak_learning_rate=learning_rate,
        num_fine_tune_steps=num_fine_tune_steps,
    )

    logger.info(f"Training resume from {meta:,}")

    # Training.
    logger.info("\n--- Starting Training ---")

    step = initial_train_itts

    # Setup batch.
    jnp_inputs, jnp_labels = setup_batch(
        dataset,
        encoder,
        train_idx,
        num_fine_tune_steps,
        batch_size,
        step,
    )

    # Train sweep.
    state, losses = train_sweep_steps(
        train_step_auto_regression,
        state,
        jnp_inputs,
        jnp_labels,
    )

    test_loss = get_test_loss(
        state,
        test_input,
        test_label,
        vocab_size,
    )

    write_loss(
        save_path / "test_losses.csv",
        initial_train_itts,
        float(test_loss),
    )

    del losses
    del test_loss

    sample_model(
        encoder,
        state,
        seq_len,
        save_path,
        gen_its=16,
        samps_to_gen=1_000,
        tag=f"{initial_train_itts:,}",
    )

    logger.info("\n Training finished.")


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

    embs = {"xs": 256, "s": 384, "m": 512, "l": 768, "xl": 1024}
    blocks = {"xs": 4, "s": 6, "m": 12, "l": 16, "xl": 24}
    heads = {"xs": 4, "s": 6, "m": 8, "l": 12, "xl": 16}
    # itts = {"xs": 10_000, "s": 40_000, "m": 110_000, "l": 300_000, "xl": 300_000}
    itts = {"xs": 96_000, "s": 160_000, "m": 192_000, "l": 320_000, "xl": 320_000}
    sweeps = {"xs": 1_500, "s": 5_000, "m": 6_000, "l": 10_000, "xl": 10_000}

    sizes = ["xs"]
    for size in sizes:
        # Setup.
        emb = embs[size]
        block = blocks[size]
        head = heads[size]

        processed_data_home = data_root / "input_data" / "dehydration" / "processed"
        data_path = processed_data_home / "spheres_15_170m.hdf5"

        save_path = (
            data_root
            / "results"
            / "sgd_models_dehydration"
            / "scale"
            / f"{size}"
            / f"{lr}"
            / f"spheres_{emb}emb_{block}block_{head}head_15tet"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directoy: {save_path.resolve()}")
        write_stat(save_path / "stats.txt", "size:", size)

        model_setup = setup_model(
            data_path=data_path,
            d_model=emb,
            num_layers=block,
            num_heads=head,
            num_test_samps=16_000,
        )

        # Train model.
        tic = time.time()
        save_itts = train_model(
            save_path,
            batch_size=512,
            num_train_steps=itts[size],
            sweep=sweeps[size],
            learning_rate=lr,
            resume=True,
            resume_from=48_000,
            model_setup=model_setup,
            save_at=[32],
        )
        toc = time.time()

        train_time = toc - tic
        logger.info(f"Training time: {train_time:.2f} seconds")

        # Fine tune model
        tic = time.time()
        for initial_train_itts in save_itts:
            logger.info(f"Fine tuning at {initial_train_itts}")
            fine_tune_model(
                save_path,
                initial_train_itts,
                batch_size=512,
                num_fine_tune_steps=int(0.05 * initial_train_itts),
                learning_rate=lr,
                model_setup=model_setup,
            )
        toc = time.time()

        tune_time = toc - tic
        logger.info(f"Fine tune time: {tune_time:.2f} seconds")

        message = f"Fine tune time: {tune_time:.2f} seconds."
        send_ntfy(
            "usyd-knottedness",
            f"Finished training for size={size}.",
            message,
        )


if __name__ == "__main__":
    # if "scale_xlo" in sys.argv:
    #     main_train_scale(1e-4)
    if "scale_low" in sys.argv:
        main_train_scale(1e-3)
    if "scale_med" in sys.argv:
        main_train_scale(3e-3)
    # if "scale_high" in sys.argv:
    #     main_train_scale(1e-2)
