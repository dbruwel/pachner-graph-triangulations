"""Experiment to randomly search for a triangulation with all edges knotted.

For some time, it was an open conjecture as to if there could exist a single vertex triangulation
of the 3-sphere with all the edges knotted. This was resolved by Burton, and acts as a "hard"
search problem. Random search is unlikely to succeed, but it serves as a good "goal".
"""

import logging
import multiprocessing
from datetime import datetime

from pachner_traversal.classical_search.mcmc import iterate
from pachner_traversal.math.potential_functions import check_all_knotted

logger = logging.getLogger(__name__)


def sample_chain(
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
    chain_id: int,
):
    """Runs a chain of MCMC, check each new triangulation to see if it contains all knotted edges.

    Parameters
    ----------
    seed : str
        The seed triangulation for the MCMC run.
    gamma_ : float
        The random walk growth parameter.
    itts : int
        The number of iterations.
    steps : int
        The number of steps per iteration.
    chain_id : int
        The chain ID for logging purposes.

    Returns
    -------
    list
        A list of all the triangulations found that have all their edges knotted.
    """
    isos = [seed]
    all_knotted_list = []

    for itt in range(itts):
        if ((itt % 1_000 == 0) or (itt < 100 and itt % 10 == 0)) and (chain_id == 0):
            logger.info(
                f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f} at {datetime.now().strftime('%H:%M:%S')}"
            )
        current_iso = isos[-1]

        proposed_iso = iterate(current_iso, gamma_, steps=steps)
        if not proposed_iso in isos:
            all_knotted = check_all_knotted(proposed_iso)
            if all_knotted:
                logger.error(f"Chain {chain_id}: Found all knotted!")
                logger.error(f"Chain {chain_id}: IsoSig: {proposed_iso}")
                all_knotted_list.append(proposed_iso)
        isos.append(proposed_iso)

    return all_knotted_list


def run_chains(
    num_chains: int,
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
) -> list[list[str]]:
    """Runs multiple MCMC chains searching for an all unknotted edge triangulation.

    Parameters
    ----------
    num_chains : int
        The number of chains to run.
    seed : str
        The seed triangulation for each chain.
    gamma_ : float
        The random walk growth parameter.
    itts : int
        The number of iterations per chain.
    steps : int
        The number of steps per iteration.

    Returns
    -------
    list[list[str]]
        A list of lists, each containing a list of the all knotted edge triangulations.
    """
    logger.info(f"Running {num_chains:,.0f} chains with {itts:,.0f} iterations each.")

    with multiprocessing.Pool(processes=num_chains) as pool:
        args = [
            (
                seed,
                gamma_,
                itts,
                steps,
                chain_id,
            )
            for chain_id in range(num_chains)
        ]

        all_knotted_list = pool.starmap(sample_chain, args)

    return all_knotted_list


def main():
    logging.basicConfig(level=logging.INFO)
    num_chains = 7
    seed = "cMcabbgqs"
    gamma_ = 1 / 10
    itts = 10_000_000
    steps = 1

    all_knotted_list = run_chains(
        num_chains=num_chains,
        seed=seed,
        gamma_=gamma_,
        itts=itts,
        steps=steps,
    )

    logger.info(all_knotted_list)


if __name__ == "__main__":
    main()
