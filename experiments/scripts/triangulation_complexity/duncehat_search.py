import logging
import multiprocessing
from datetime import datetime

from regina import Triangulation3

from pachner_traversal.mcmc import iterate
from pachner_traversal.potential_functions import check_all_unknotted, is_knotted

logger = logging.getLogger(__name__)


def check_contains_dunce_hat(
    iso: str,
) -> bool:
    t = Triangulation3(iso)
    faces = [f for f in t.faces(2)]
    for face in faces:
        if face.type() == face.DUNCEHAT:
            return True

    return False


def check_dunce_hat_and_knotted(iso: str) -> bool:
    t = Triangulation3(iso)
    faces = [f for f in t.faces(2)]
    for face in faces:
        if face.type() == face.DUNCEHAT:
            edge_id = face.edge(0).index()
            knot = Triangulation3(iso)
            knot.pinchEdge(knot.edges()[edge_id])
            if is_knotted(knot):
                return True

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
            # Code to check for a dunce hat where all edges are not knotted
            # Run cheap check first
            # contains_dunce_hat = check_contains_dunce_hat(proposed_iso)
            # if contains_dunce_hat:
            #     all_unknotted = check_all_unknotted(proposed_iso)
            #     if all_unknotted:
            #         logger.info("Found all unknotted with dunce hat!")
            #         logger.info(f"IsoSig: {proposed_iso}")

            # Code to check for a duncehat where dunce hat edges are knotted
            dunce_hat_and_knotted = check_dunce_hat_and_knotted(proposed_iso)
            if dunce_hat_and_knotted:
                logger.info("Found dunce hat and knotted!")
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
