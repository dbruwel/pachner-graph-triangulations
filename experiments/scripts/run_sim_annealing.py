import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import pachner_traversal.potential_functions as potentials
from pachner_traversal.simulated_annealing import run_chains
from pachner_traversal.utils import results_path


def run_sim_annealing(
    res_path: Path,
    potential: Callable[
        [str], tuple[float | np.floating[Any], float | np.floating[Any], int, bool]
    ],
) -> None:
    path = results_path(res_path)

    res = run_chains(
        betas=[1] * 6,
        potential=potential,
        seed="cMcabbgqs",
        gamma_=0.2,
        itts=10_000,
        steps=10,
        lambda_=1e-4,
        alpha=1e-4,
        target_acceptance=0.2,
    )

    isos_lists = res["isos_lists"]
    betas_lists = res["betas_lists"]
    acceptances_lists = res["acceptances_lists"]
    scores = res["final_scores"]
    pns = res["final_pns"]
    unknotted = res["final_counts_unknotted"]

    scores_df = pd.DataFrame.from_dict(scores, orient="index", columns=["score"])
    pn_df = pd.DataFrame.from_dict(pns, orient="index", columns=["pn"])
    unknot_df = pd.DataFrame.from_dict(unknotted, orient="index", columns=["unknotted"])
    isos_df = pd.DataFrame(isos_lists).T
    betas_df = pd.DataFrame(betas_lists).T
    acceptances_df = pd.DataFrame(acceptances_lists).T

    scores_df.to_csv(f"{path}/scores.csv")
    pn_df.to_csv(f"{path}/pns.csv")
    unknot_df.to_csv(f"{path}/counts_unknotted.csv")
    isos_df.to_csv(f"{path}/isos.csv", index=False)
    betas_df.to_csv(f"{path}/betas.csv", index=False)
    acceptances_df.to_csv(f"{path}/acceptances.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)

    logger.info("Degree of Alexander polynomial potential")
    run_sim_annealing(
        Path("sim_annealing") / "degree_alexander_polynomial",
        potentials.Potential(
            potentials.DegreeAlexanderPolynomial, max_size=30
        ).calc_potential,
    )

    logger.info("")
    logger.info("")
    logger.info("Determinant of Alexander polynomial potential")
    run_sim_annealing(
        Path("sim_annealing") / "determinant_alexander_polynomial",
        potentials.Potential(
            potentials.DeterminantAlexanderPolynomial, max_size=30
        ).calc_potential,
    )

    logger.info("")
    logger.info("")
    logger.info("Norm of Alexander polynomial potential")
    run_sim_annealing(
        Path("sim_annealing") / "norm_alexander_polynomial",
        potentials.Potential(
            potentials.NormAlexanderPolynomial, max_size=30
        ).calc_potential,
    )

    logger.info("")
    logger.info("")
    logger.info("Number of generators potential")
    run_sim_annealing(
        Path("sim_annealing") / "number_of_generators",
        potentials.Potential(potentials.NumGenerators, max_size=30).calc_potential,
    )

    logger.info("")
    logger.info("")
    logger.info("Variance of edge degree potential")
    run_sim_annealing(
        Path("sim_annealing") / "variance_edge_degree",
        potentials.Potential(potentials.VarianceEdgeDegree, max_size=30).calc_potential,
    )
