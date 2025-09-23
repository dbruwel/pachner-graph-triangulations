import os
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt

data_path = pathlib.Path(__file__).parent.parent.parent / "data"


def results_path(res_name: str | pathlib.Path) -> pathlib.Path:
    res_path = data_path / "results" / res_name

    path = res_path / datetime.now().strftime("%Y%m%d_%H%M")
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_style() -> None:
    style_path = pathlib.Path(__file__).parent / "stylelib" / "journal.mplstyle"

    plt.style.use(style_path)
