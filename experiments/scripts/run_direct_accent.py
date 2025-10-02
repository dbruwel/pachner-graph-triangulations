import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import pachner_traversal.potential_functions as potentials
from pachner_traversal.direct_ascent import run_accent
from pachner_traversal.utils import results_path

logging.basicConfig(level=logging.INFO)


def run_direct_accent(
    res_path: Path,
    potential: Callable[
        [str], tuple[float | np.floating[Any], float | np.floating[Any], int, bool]
    ],
) -> None:
    base_isos = ["cMcabbgqs"] * 5
    betas = [10] * 5

    res = run_accent(
        base_isos=base_isos,
        potential=potential,
        betas=betas,
        height=30,
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

    path = results_path(res_path)

    df_isos.to_csv(path / "isos.csv", index=False)
    df_scores.to_csv(path / "scores.csv", index=False)
    df_avg_scores.to_csv(path / "avg_scores.csv", index=False)
    df_p_knotteds.to_csv(path / "p_knotteds.csv", index=False)
    df_count_unknotteds.to_csv(path / "count_unknotteds.csv", index=False)
    df_all_knotteds.to_csv(path / "all_knotteds.csv", index=False)
    df_betas.to_csv(path / "betas.csv", index=False)


if __name__ == "__main__":
    run_direct_accent(
        Path("direct_ascent") / "degree_alexander_polynomial",
        potentials.Potential(
            potentials.DegreeAlexanderPolynomial, max_size=None
        ).calc_potential,
    )

    run_direct_accent(
        Path("direct_ascent") / "determinant_alexander_polynomial",
        potentials.Potential(
            potentials.DeterminantAlexanderPolynomial, max_size=None
        ).calc_potential,
    )

    run_direct_accent(
        Path("direct_ascent") / "norm_alexander_polynomial",
        potentials.Potential(
            potentials.NormAlexanderPolynomial, max_size=None
        ).calc_potential,
    )

    run_direct_accent(
        Path("direct_ascent") / "number_of_generators",
        potentials.Potential(potentials.NumGenerators, max_size=None).calc_potential,
    )

    run_direct_accent(
        Path("direct_ascent") / "variance_edge_degree",
        potentials.Potential(
            potentials.VarianceEdgeDegree, max_size=None
        ).calc_potential,
    )
