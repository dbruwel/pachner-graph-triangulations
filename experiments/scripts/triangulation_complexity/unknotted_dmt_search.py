import logging
import multiprocessing
from datetime import datetime

from regina import Triangulation3

from pachner_traversal.mcmc import iterate
from pachner_traversal.potential_functions import check_all_unknotted
from pachner_traversal.dmt import estimate_critical_count

logger = logging.getLogger(__name__)


def check_perfect_critical(iso: str, base_itts=10, total_itts=10) -> bool:
    min_critical = float("inf")
    for _ in range(total_itts):
        critical = estimate_critical_count(iso, itts=base_itts)
        if critical == 2:
            return True
        if critical < min_critical:
            min_critical = critical

    return False


def sample_chain(
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
    chain_id: int,
):
    isos = [seed]

    for itt in range(itts):
        if (itt % 1_000 == 0) or (itt < 100 and itt % 10 == 0):
            if chain_id == 0:
                logger.info(
                    f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f} at {datetime.now().strftime('%H:%M:%S')}"
                )
        current_iso = isos[-1]

        proposed_iso = iterate(current_iso, gamma_, steps=steps)
        if not proposed_iso in isos:
            all_unknotted = check_all_unknotted(proposed_iso)
            if all_unknotted:
                perfect_critical = check_perfect_critical(proposed_iso)
                if not perfect_critical:
                    logger.info("Found triangulation with non-perfect critical count!")
                    logger.info(f"IsoSig: {proposed_iso}")
                else:
                    logger.info("Found triangulation with perfect critical count!")
                    logger.info(f"IsoSig: {proposed_iso}")

        isos.append(proposed_iso)

    return []


def run_chains(
    num_chains: int,
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
) -> list[list[str]]:
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

        isos_lists = pool.starmap(sample_chain, args)

    return isos_lists


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    run_chains(
        num_chains=1,
        seed="cMcabbgqs",
        gamma_=1 / 10,
        itts=10_000,
        steps=100,
    )
