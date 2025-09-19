import os
import pathlib
from datetime import datetime


def results_path(res_name):
    src = os.path.abspath(__file__)
    res_path = pathlib.Path(src).parent.parent.parent / "data" / "results" / res_name

    path = res_path / datetime.now().strftime("%Y%m%d_%H%M")
    path.mkdir(parents=True, exist_ok=True)
    return path
