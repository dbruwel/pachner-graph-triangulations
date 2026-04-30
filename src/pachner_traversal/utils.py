import os
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import requests

data_root = pathlib.Path(__file__).parent.parent.parent / "data"

leading_chars = {
    10: "k",
    11: "l",
    12: "m",
    13: "n",
    14: "o",
    15: "p",
    16: "q",
    17: "r",
    18: "s",
    19: "t",
    20: "u",
}


def create_results_path(res_name: str | pathlib.Path) -> pathlib.Path:
    res_path = data_root / "results" / res_name

    path = res_path / datetime.now().strftime("%Y%m%d_%H%M")
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_style() -> None:
    style_path = pathlib.Path(__file__).parent / "stylelib" / "journal.mplstyle"

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
