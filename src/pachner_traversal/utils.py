import csv
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml

logger_config = {
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "level": logging.DEBUG,
}


def get_data_root(nci: bool = False) -> Path:
    if nci:
        return Path("/g/data/io00/js1886/pachner-graph-triangulations/data")
    else:
        return Path(__file__).parent.parent.parent / "data"


def create_results_path(
    res_name: str | Path,
    data_root: Path,
) -> Path:
    res_path = data_root / "results" / res_name

    path = res_path / datetime.now().strftime("%Y%m%d_%H%M")
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_style() -> None:
    style_path = Path(__file__).parent / "stylelib" / "journal.mplstyle"

    plt.style.use(style_path)


def compute_rhat(df):
    L = df.shape[0]  # Number of samples per chain
    J = df.shape[1]  # Number of chains
    S = L * J  # Total number of samples

    chain_mean = df.mean()
    B = L * chain_mean.var()
    W = df.var().mean()

    R = (W * (L - 1) / L + B / L) / W
    r = np.sqrt(R)
    ESS = S / R

    return r, ESS


def to_numpy(M):
    rows = M.rows()
    cols = M.columns()
    A = np.zeros((rows, cols), dtype=int)
    for i in range(rows):
        for j in range(cols):
            A[i, j] = M.entry(i, j).longValue()

    return A


def send_ntfy(topic: str, title: str, message: str) -> None:
    url = f"https://ntfy.sh/{topic}"
    headers = {"Title": title, "Priority": "high", "Tags": "python,snake"}

    requests.post(url, data=message.encode("utf-8"), headers=headers)


def write_loss(save_path, step, loss):
    if isinstance(loss, (float, np.floating, int, np.integer)):
        with open(save_path, "a") as f:
            f.write(f"{step},{loss}\n")
    else:
        with open(save_path, "a") as f:
            for st, lo in zip(step, loss):
                f.write(f"{st},{lo}\n")


def save_model(save_path, state, tag=None):
    fname = f"params_{tag}.pkl" if tag else "params.pkl"

    with open(save_path / fname, "wb") as file:
        pickle.dump(state.params, file)


def load_model(save_path, fname):
    with open(save_path / fname, "rb") as file:
        params = pickle.load(file)

    return params


def write_stat(stat_file_path, stat_name, stat_value):
    with open(stat_file_path, "a") as f:
        f.write(f"{stat_name}, {stat_value}\n")


def create_sample_schedule(batch_size, dataset_size, epochs, num_itts, seed=42):
    np.random.seed(seed)

    num_samples = num_itts * batch_size
    num_unique_samples = num_samples // epochs

    if num_samples % epochs != 0:
        raise ValueError("Total samples is not a multiple of epochs.")
    if num_unique_samples > dataset_size:
        raise ValueError("Total unique samples exceeds dataset size.")

    unique_samples = np.random.choice(
        dataset_size, size=num_unique_samples, replace=False
    )

    schedule = np.concatenate(
        [np.random.permutation(unique_samples) for _ in range(epochs)]
    )

    return schedule


def get_sample_idx(schedule, batch_size, itt):
    start_pos = itt * batch_size
    end_pos = start_pos + batch_size
    idxs = schedule[start_pos:end_pos]

    return idxs


def get_random_sample_idx(batch_size, dataset_size):
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


def normalize(x):
    return (x - x.mean()) / x.std()


def silence_jax():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)


def read_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_to_raw_indices(train_batch_idx, test_indices_sorted):
    train_batch = np.asarray(train_batch_idx)
    test_idxs = np.asarray(test_indices_sorted)

    raw_batch = train_batch.copy()
    k = np.searchsorted(test_idxs, raw_batch, side="right")

    while True:
        next_raw_batch = train_batch + k
        if np.array_equal(next_raw_batch, raw_batch):
            break

        raw_batch = next_raw_batch
        k = np.searchsorted(test_idxs, raw_batch, side="right")

    return raw_batch


def name_to_fname(name):
    return name.lower().replace(" ", "_")
