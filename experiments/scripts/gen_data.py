import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pachner_traversal.mcmc import iterate
from pachner_traversal.potential_functions import calc_composite_potential
from pachner_traversal.utils import data_path

logger = logging.getLogger(__name__)


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
        isos.append(proposed_iso)

    return isos


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


def run_mcmc(
    num_chains: int,
    gamma_: float,
    itts: int,
    steps: int,
) -> pd.DataFrame:
    results = run_chains(
        num_chains=num_chains,
        seed="cMcabbgqs",
        gamma_=gamma_,
        itts=itts,
        steps=steps,
    )

    isos_lists_df = pd.DataFrame(results).T
    return isos_lists_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_path = data_path / "input_data" / "dehydration" / "raw" / "mcmc_samples"

    isos_lists_df = run_mcmc(
        num_chains=30,
        gamma_=1 / 10,
        itts=100_000,
        steps=1,
    )

    isos_list = isos_lists_df.to_numpy().flatten()
    isos_list = isos_list.astype(str)
    isos_list = np.unique(isos_list)

    save_path.mkdir(parents=True, exist_ok=True)
    save_file = save_path / "mcmc_samples.csv"
    np.savetxt(save_file, isos_list, fmt="%s")
