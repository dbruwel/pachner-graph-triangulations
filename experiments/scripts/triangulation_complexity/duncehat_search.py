"""Experiment to search for a triangulation with a knotted dunce hat.

This is an experiment script that performs MCMC sampling of triangulations, and on each new sample
checks if the sample containsa a dunce hat, and if it does, checks if the dunce hat edge is knotted.
There is only one edge in a dunce hat, so only the first edge needs to be checked. The conjecture is
that because a duncehat is contractible, the edge will be unknotted, that is to say, the dunce hat
essentially forms a disk. This experiment is to ensure that this logic is not "obviosuly" wrong before
further theorey is attempted.

The code is set to run in parallel.
"""

import logging
import multiprocessing
from datetime import datetime

from regina import Triangulation3

from pachner_traversal.classical_search.mcmc import iterate
from pachner_traversal.math.potential_functions import is_knotted

logger = logging.getLogger(__name__)


def check_contains_dunce_hat(
    iso: str,
) -> bool:
    """Checks if a triangulation (given by an IsoSig) contains an dunce hat.

    Parameters
    ----------
    iso : str
        The triangulation isomorphism signature.

    Returns
    -------
    bool
        If there is a dunce hat or not.
    """
    t = Triangulation3(iso)
    faces = [f for f in t.faces(2)]
    for face in faces:
        if face.type() == face.DUNCEHAT:
            return True

    return False


def check_dunce_hat_and_knotted(iso: str) -> bool:
    """Check if a given triangulation (given by an IsoSig) contains a knotted dunce hat.
    That is a dunce hat where its edge forms a knot in the triangulated 3-sphere.

    Parameters
    ----------
    iso : str
        The triangulation isomorphism signature.

    Returns
    -------
    bool
        If there is a knotted dunce hat in the triangulation.
    """
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
) -> list:
    """Samples a chain of triangulations, check if there is a knotted dunce hat on each new sample.
    If there is, it is logged, and the script continues.

    Parameters
    ----------
    seed : str
        The starting seed triangulation.
    gamma_ : float
        The random walk control parameter.
    itts : int
        The number of iterations to run.
    steps : int
        The number of steps of MCMC per iteration.
    chain_id : int
        An ID for the chain for logging purposes in parallel runs.

    Returns
    -------
    list
        An empty list for compatibility purposes.
    """
    isos = [seed]

    for itt in range(itts):
        if ((itt % 1_000 == 0) or (itt < 100 and itt % 10 == 0)) and (chain_id == 0):
            msg = f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f} at {datetime.now().strftime('%H:%M:%S')}"
            logger.info(msg)
        current_iso = isos[-1]

        proposed_iso = iterate(current_iso, gamma_, steps=steps)
        if not proposed_iso in isos:
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
    """Runs multiple MCMC chains in parallel searching for a knotted dunce hat.

    Parameters
    ----------
    num_chains : int
        The number of chains to run in parllel
    seed : str
        The starting seed for each chain.
    gamma_ : float
        The random walk control parameter.
    itts : int
        The number of iterations per chain.
    steps : int
        The number of MCMC steps per chain.

    Returns
    -------
    list[list[str]]
        A list of empty lists, one per chain, for compatibility reasons.
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

        isos_lists = pool.starmap(sample_chain, args)

    return isos_lists


def main():
    logging.basicConfig(level=logging.INFO)
    num_chains = 1
    seed = "cMcabbgqs"
    gamma_ = 1 / 10
    itts = 10_000
    steps = 100

    run_chains(
        num_chains=num_chains,
        seed=seed,
        gamma_=gamma_,
        itts=itts,
        steps=steps,
    )


if __name__ == "__main__":
    main()
