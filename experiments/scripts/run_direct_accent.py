import logging
from pathlib import Path

import numpy as np
import pandas as pd

import pachner_traversal.potential_functions as potentials
from pachner_traversal.data_io import Dataset
from pachner_traversal.direct_ascent import run_accent
from pachner_traversal.utils import data_path, results_path

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    dataset = Dataset(
        data_path / "input_data" / "processed" / "spheres_5tet.hdf5", num_test_samps=0
    )

    # base_isos = dataset.samp_batch(batch_size=100, replace=False)
    base_isos = ["cMcabbgqs"] * 500
    betas = (
        [0.1] * 100
        + [np.sqrt(0.1)] * 100
        + [1] * 100
        + [np.sqrt(10)] * 100
        + [10] * 100
    )

    # betas = [10] * len(base_isos)

    potential = potentials.Potential(
        potentials.DegreeAlexanderPolynomial, max_size=None
    ).calc_potential

    res = run_accent(
        base_isos=base_isos,
        potential=potential,
        betas=betas,
        height=10,
    )

    isos_lists = res
    isos_lists = [isos_list for isos_list in isos_lists if isos_list is not None]

    df_isos = pd.DataFrame([isos_list[0] for isos_list in isos_lists])
    df_scores = pd.DataFrame([isos_list[1] for isos_list in isos_lists])
    df_avg_scores = pd.DataFrame([isos_list[2] for isos_list in isos_lists])
    df_p_knotteds = pd.DataFrame([isos_list[3] for isos_list in isos_lists])
    df_count_unknotteds = pd.DataFrame([isos_list[4] for isos_list in isos_lists])
    df_all_knotteds = pd.DataFrame([isos_list[5] for isos_list in isos_lists])
    df_betas = pd.DataFrame(
        [isos_list[6] for isos_list in isos_lists], columns=["beta"]
    )

    path = results_path(Path("direct_ascent") / "degree_alexander_polynomial")

    df_isos.to_csv(path / "isos.csv", index=False)
    df_scores.to_csv(path / "scores.csv", index=False)
    df_avg_scores.to_csv(path / "avg_scores.csv", index=False)
    df_p_knotteds.to_csv(path / "p_knotteds.csv", index=False)
    df_count_unknotteds.to_csv(path / "count_unknotteds.csv", index=False)
    df_all_knotteds.to_csv(path / "all_knotteds.csv", index=False)
    df_betas.to_csv(path / "betas.csv", index=False)
