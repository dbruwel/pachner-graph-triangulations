import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from regina import Triangulation3

from pachner_traversal.mcmc import iterate
from pachner_traversal.utils import results_path
from pachner_traversal.potential_functions import check_all_unknotted, check_all_knotted

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


def sample_chain(
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
    chain_id: int,
):
    isos = [seed]
    all_knotted_list = []

    for itt in range(itts):
        if (itt % 1_000 == 0) or (itt < 100 and itt % 10 == 0):
            if chain_id == 0:
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
                raise ValueError("Found all knotted!")
        isos.append(proposed_iso)

    return all_knotted_list


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

        all_knotted_list = pool.starmap(sample_chain, args)

    return all_knotted_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    all_knotted_list = run_chains(
        num_chains=7,
        seed="cMcabbgqs",
        gamma_=1 / 10,
        itts=10_000_000,
        steps=1,
    )

    print(all_knotted_list)
