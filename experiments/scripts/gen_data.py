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
        if proposed_iso[0] == "v":
            continue
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

    for i in range(10):
        logger.info(
            f"Starting MCMC run {i + 1}/10 at {datetime.now().strftime('%H:%M:%S')}"
        )
        isos_lists_df = run_mcmc(
            num_chains=30,
            gamma_=1 / 5,
            itts=1_000_000,
            steps=1,
        )

        isos_list = isos_lists_df.to_numpy().flatten()
        isos_list = isos_list.astype(str)
        isos_list = np.unique(isos_list)

        logger.info(f"Total unique samples: {len(isos_list):,}.")
        logger.info(f"Example samples: {isos_list[-5:]}")

        samps10 = isos_list[np.char.startswith(isos_list, "k")]
        samps11 = isos_list[np.char.startswith(isos_list, "l")]
        samps12 = isos_list[np.char.startswith(isos_list, "m")]
        samps13 = isos_list[np.char.startswith(isos_list, "n")]
        samps14 = isos_list[np.char.startswith(isos_list, "o")]
        samps15 = isos_list[np.char.startswith(isos_list, "p")]
        samps16 = isos_list[np.char.startswith(isos_list, "q")]
        samps17 = isos_list[np.char.startswith(isos_list, "r")]
        samps18 = isos_list[np.char.startswith(isos_list, "s")]
        samps19 = isos_list[np.char.startswith(isos_list, "t")]
        samps20 = isos_list[np.char.startswith(isos_list, "u")]

        logger.info(f"{len(samps10):,} samples for N=10.")
        with open(save_path / "samps10.txt", "a") as f:
            np.savetxt(f, samps10, fmt="%s")
        logger.info(f"{len(samps11):,} samples for N=11.")
        with open(save_path / "samps11.txt", "a") as f:
            np.savetxt(f, samps11, fmt="%s")
        logger.info(f"{len(samps12):,} samples for N=12.")
        with open(save_path / "samps12.txt", "a") as f:
            np.savetxt(f, samps12, fmt="%s")
        logger.info(f"{len(samps13):,} samples for N=13.")
        with open(save_path / "samps13.txt", "a") as f:
            np.savetxt(f, samps13, fmt="%s")
        logger.info(f"{len(samps14):,} samples for N=14.")
        with open(save_path / "samps14.txt", "a") as f:
            np.savetxt(f, samps14, fmt="%s")
        logger.info(f"{len(samps15):,} samples for N=15.")
        with open(save_path / "samps15.txt", "a") as f:
            np.savetxt(f, samps15, fmt="%s")
        logger.info(f"{len(samps16):,} samples for N=16.")
        with open(save_path / "samps16.txt", "a") as f:
            np.savetxt(f, samps16, fmt="%s")
        logger.info(f"{len(samps17):,} samples for N=17.")
        with open(save_path / "samps17.txt", "a") as f:
            np.savetxt(f, samps17, fmt="%s")
        logger.info(f"{len(samps18):,} samples for N=18.")
        with open(save_path / "samps18.txt", "a") as f:
            np.savetxt(f, samps18, fmt="%s")
        logger.info(f"{len(samps19):,} samples for N=19.")
        with open(save_path / "samps19.txt", "a") as f:
            np.savetxt(f, samps19, fmt="%s")
        logger.info(f"{len(samps20):,} samples for N=20.")
        with open(save_path / "samps20.txt", "a") as f:
            np.savetxt(f, samps20, fmt="%s")

        save_path.mkdir(parents=True, exist_ok=True)
        save_file = save_path / "mcmc_samples.txt"
        with open(save_path / "mcmc_samples.txt", "a") as f:
            np.savetxt(f, isos_list, fmt="%s")
