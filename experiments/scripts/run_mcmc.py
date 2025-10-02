import logging
import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

import pachner_traversal.potential_functions as potentials
from pachner_traversal.mcmc import iterate
from pachner_traversal.utils import results_path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def sample_chain(
    scores: dict[str, float | np.floating],
    pns: dict[str, float | np.floating],
    counts_unknotted: dict[str, int],
    potential: Callable[
        [str], tuple[float | np.floating, float | np.floating, int, bool]
    ],
    seed: str = "cMcabbgqs",
    gamma_: float = 1 / 5,
    itts: int = 1_000,
    steps: int = 1,
    chain_id: Optional[int] = None,
):
    isos = [seed]
    seed_score, seed_pn, seed_count_unknotted, _ = potential(seed)

    scores[seed] = seed_score
    pns[seed] = seed_pn
    counts_unknotted[seed] = seed_count_unknotted

    for itt in range(itts):
        if itt % 10_000 == 0:
            if chain_id == 0:
                logger.info(f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f}")
        current_iso = isos[-1]

        proposed_iso = iterate(current_iso, gamma_, steps=steps)

        with multiprocessing.Lock() as lock:
            if proposed_iso in scores:
                proposed_score = scores[proposed_iso]
            else:
                try:
                    proposed_score, proposed_pn, count_unknotted, _ = potential(
                        proposed_iso
                    )
                    scores[proposed_iso] = proposed_score
                    pns[proposed_iso] = proposed_pn
                    counts_unknotted[proposed_iso] = count_unknotted

                except Exception as e:
                    logger.error(
                        f"Error processing iso {proposed_iso} in score computation. Skipping."
                    )
                    logger.error(f"Exception: {e}")
                    continue

        isos.append(proposed_iso)

    return isos


def run_chains(
    num_chains: int,
    potential: Callable[
        [str], tuple[float | np.floating, float | np.floating, int, bool]
    ],
    seed: str = "cMcabbgqs",
    gamma_: float = 1 / 5,
    itts: int = 1_000,
    steps: int = 1,
) -> dict[str, Any]:
    logger.info(f"Running {num_chains} chains with {itts} iterations each.")

    with multiprocessing.Manager() as manager:
        scores = manager.dict()
        pns = manager.dict()
        counts_unknotted = manager.dict()

        with multiprocessing.Pool(processes=num_chains) as pool:
            args = [
                (
                    scores,
                    pns,
                    counts_unknotted,
                    potential,
                    seed,
                    gamma_,
                    itts,
                    steps,
                    chain_id,
                )
                for chain_id in range(num_chains)
            ]

            isos_lists = pool.starmap(sample_chain, args)

        final_scores = dict(scores)
        final_pns = dict(pns)
        final_num_unk = dict(counts_unknotted)

    return {
        "isos_lists": isos_lists,
        "final_scores": final_scores,
        "final_pns": final_pns,
        "final_num_unk": final_num_unk,
    }


def run_mcmc(
    res_path: Path,
    potential: Callable[
        [str], tuple[float | np.floating[Any], float | np.floating[Any], int, bool]
    ],
) -> None:
    path = results_path(res_path)

    results = run_chains(
        num_chains=7,
        potential=potential,
        seed="cMcabbgqs",
        itts=10_000,
        steps=1,
    )

    isos_lists_df = pd.DataFrame(results["isos_lists"]).T
    final_scores_df = pd.Series(results["final_scores"]).rename("score")
    final_pns_df = pd.Series(results["final_pns"]).rename("p_n")
    final_final_num_unk_df = pd.Series(results["final_num_unk"]).rename("num_unk")

    isos_lists_df.to_csv(path / "isos_lists.csv", index=False)
    final_scores_df.to_csv(path / "final_scores.csv", index=True)
    final_pns_df.to_csv(path / "final_pns.csv", index=True)
    final_final_num_unk_df.to_csv(path / "final_num_unk.csv", index=True)


if __name__ == "__main__":
    run_mcmc(
        Path("mcmc") / "degree_alexander_polynomial",
        potentials.Potential(
            potentials.DegreeAlexanderPolynomial, max_size=None
        ).calc_potential,
    )

    run_mcmc(
        Path("mcmc") / "determinant_alexander_polynomial",
        potentials.Potential(
            potentials.DeterminantAlexanderPolynomial, max_size=None
        ).calc_potential,
    )

    run_mcmc(
        Path("mcmc") / "norm_alexander_polynomial",
        potentials.Potential(
            potentials.NormAlexanderPolynomial, max_size=None
        ).calc_potential,
    )

    run_mcmc(
        Path("mcmc") / "number_of_generators",
        potentials.Potential(potentials.NumGenerators, max_size=None).calc_potential,
    )

    run_mcmc(
        Path("mcmc") / "variance_edge_degree",
        potentials.Potential(
            potentials.VarianceEdgeDegree, max_size=None
        ).calc_potential,
    )
