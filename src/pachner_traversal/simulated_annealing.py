import logging
import multiprocessing
from collections.abc import Callable
from typing import Optional, Any

import numpy as np
import regina

from .mcmc import iterate

logger = logging.getLogger(__name__)


def sample_chain(
    beta: float,
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
    lambda_: float = 1 / 5,
    alpha: float = 1 / 5,
    target_acceptance: float = 0.2,
):
    isos = [seed]
    seed_score, seed_pn, seed_count_unknotted, _ = potential(seed)

    scores[seed] = seed_score
    pns[seed] = seed_pn
    counts_unknotted[seed] = seed_count_unknotted

    moving_average = target_acceptance
    betas = [beta]
    acceptances = [1]

    for itt in range(itts):
        if itt % 1_000 == 0:
            if chain_id == 0:
                logger.info(f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f}")
        current_iso = isos[-1]

        with multiprocessing.Lock() as lock:
            current_score = scores[current_iso]

        proposed_iso = iterate(current_iso, gamma_, steps=steps)
        if proposed_iso == current_iso:
            isos.append(proposed_iso)
            moving_average = (1 - alpha) * moving_average + alpha * 0.0
            acceptances.append(0)
        else:
            with multiprocessing.Lock() as lock:
                if proposed_iso in scores:
                    proposed_score = scores[proposed_iso]
                else:
                    try:
                        proposed_score, proposed_pn, count_unknotted, all_knoted = (
                            potential(proposed_iso)
                        )
                        scores[proposed_iso] = proposed_score
                        pns[proposed_iso] = proposed_pn
                        counts_unknotted[proposed_iso] = count_unknotted

                        if all_knoted:
                            logger.critical(
                                f"{proposed_iso}: All knotted, solution found. Exiting."
                            )
                            isos.append(proposed_iso)
                            break
                    except Exception as e:
                        logger.error(
                            f"Error processing iso {proposed_iso} in score computation. Skipping."
                        )
                        logger.error(f"Exception: {e}")
                        continue

            if proposed_score > current_score:
                isos.append(proposed_iso)
                moving_average = (1 - alpha) * moving_average + alpha * 1.0
                acceptances.append(1)
            else:
                try:
                    p = np.exp(-beta * (current_score - proposed_score))
                    acc_alpha = np.random.random()

                    if p > acc_alpha:
                        isos.append(proposed_iso)
                        moving_average = (1 - alpha) * moving_average + alpha * 1.0
                        acceptances.append(1)
                    else:
                        isos.append(current_iso)
                        moving_average = (1 - alpha) * moving_average + alpha * 0.0
                        acceptances.append(0)
                except Exception as e:
                    logger.error(
                        f"Error in acceptance probability calculation for iso {proposed_iso}. Skipping."
                    )
                    logger.error(f"Exception: {e}")
                    isos.append(current_iso)
                    moving_average = (1 - alpha) * moving_average + alpha * 0.0
                    acceptances.append(0)

            beta = beta * np.exp(alpha * (moving_average - target_acceptance))
            betas.append(beta)

    return isos, betas, acceptances


def run_chains(
    betas: list[float | np.floating],
    potential: Callable[
        [str], tuple[float | np.floating, float | np.floating, int, bool]
    ],
    seed: str = "cMcabbgqs",
    gamma_: float = 1 / 5,
    itts: int = 1_000,
    steps: int = 1,
    lambda_: float = 1.0,
    alpha: float = 1 / 5,
    target_acceptance: float = 0.2,
) -> dict[str, Any]:
    with multiprocessing.Manager() as manager:
        scores = manager.dict()
        pns = manager.dict()
        counts_unknotted = manager.dict()

        with multiprocessing.Pool(processes=len(betas)) as pool:
            args = [
                (
                    beta,
                    scores,
                    pns,
                    counts_unknotted,
                    potential,
                    seed,
                    gamma_,
                    itts,
                    steps,
                    chain_id,
                    lambda_,
                    alpha,
                    target_acceptance,
                )
                for chain_id, beta in enumerate(betas)
            ]

            results = pool.starmap(sample_chain, args)

        final_scores = dict(scores)
        final_pns = dict(pns)
        final_counts_unknotted = dict(counts_unknotted)

        isos_lists = [result[0] for result in results]
        betas_lists = [result[1] for result in results]
        acceptances_lists = [result[2] for result in results]

    return {
        "isos_lists": isos_lists,
        "betas_lists": betas_lists,
        "acceptances_lists": acceptances_lists,
        "final_scores": final_scores,
        "final_pns": final_pns,
        "final_counts_unknotted": final_counts_unknotted,
    }
