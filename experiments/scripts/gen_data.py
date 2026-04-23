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
        itts=1_000_000,
        steps=1,
    )

    isos_list = isos_lists_df.to_numpy().flatten()
    isos_list = isos_list.astype(str)
    samps10 = np.unique(isos_list[np.char.startswith(isos_list, "k")])
    samps11 = np.unique(isos_list[np.char.startswith(isos_list, "l")])
    samps12 = np.unique(isos_list[np.char.startswith(isos_list, "m")])
    samps13 = np.unique(isos_list[np.char.startswith(isos_list, "n")])
    samps14 = np.unique(isos_list[np.char.startswith(isos_list, "o")])
    samps15 = np.unique(isos_list[np.char.startswith(isos_list, "p")])
    samps16 = np.unique(isos_list[np.char.startswith(isos_list, "q")])
    samps17 = np.unique(isos_list[np.char.startswith(isos_list, "r")])
    samps18 = np.unique(isos_list[np.char.startswith(isos_list, "s")])
    samps19 = np.unique(isos_list[np.char.startswith(isos_list, "t")])
    samps20 = np.unique(isos_list[np.char.startswith(isos_list, "u")])

    logger.info(f"{len(samps10):,} samples for N=10.")
    np.savetxt(save_path / "samps10.txt", samps10, fmt="%s")
    logger.info(f"{len(samps11):,} samples for N=11.")
    np.savetxt(save_path / "samps11.txt", samps11, fmt="%s")
    logger.info(f"{len(samps12):,} samples for N=12.")
    np.savetxt(save_path / "samps12.txt", samps12, fmt="%s")
    logger.info(f"{len(samps13):,} samples for N=13.")
    np.savetxt(save_path / "samps13.txt", samps13, fmt="%s")
    logger.info(f"{len(samps14):,} samples for N=14.")
    np.savetxt(save_path / "samps14.txt", samps14, fmt="%s")
    logger.info(f"{len(samps15):,} samples for N=15.")
    np.savetxt(save_path / "samps15.txt", samps15, fmt="%s")
    logger.info(f"{len(samps16):,} samples for N=16.")
    np.savetxt(save_path / "samps16.txt", samps16, fmt="%s")
    logger.info(f"{len(samps17):,} samples for N=17.")
    np.savetxt(save_path / "samps17.txt", samps17, fmt="%s")
    logger.info(f"{len(samps18):,} samples for N=18.")
    np.savetxt(save_path / "samps18.txt", samps18, fmt="%s")
    logger.info(f"{len(samps19):,} samples for N=19.")
    np.savetxt(save_path / "samps19.txt", samps19, fmt="%s")
    logger.info(f"{len(samps20):,} samples for N=20.")
    np.savetxt(save_path / "samps20.txt", samps20, fmt="%s")
