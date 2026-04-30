import logging
import multiprocessing
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pachner_traversal.mcmc import iterate
from pachner_traversal.potential_functions import calc_composite_potential
from pachner_traversal.utils import results_path

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
    res_path: Path,
    num_chains: int,
    gamma_: float,
    itts: int,
    steps: int,
) -> tuple[pd.DataFrame, Path]:
    results = run_chains(
        num_chains=num_chains,
        seed="cMcabbgqs",
        gamma_=gamma_,
        itts=itts,
        steps=steps,
    )

    path = results_path(res_path)

    isos_lists_df = pd.DataFrame(results).T
    isos_lists_df.to_csv(path / "isos_lists.csv", index=False)
    return isos_lists_df, path


def get_score(
    save_path: Path, isos_list_df: pd.DataFrame, restart: bool = True
) -> pd.DataFrame | None:
    unique_isos = np.unique(isos_list_df.values.flatten())
    logger.info(
        f"Calculating composite potential for {len(unique_isos):,} unique isomorphisms."
    )
    if restart:
        logger.info(
            "Restarting calculation. Will overwrite existing scores if file is present."
        )
    else:
        try:
            existing_scores = pd.read_csv(
                save_path / "composite_scores.csv", index_col="iso"
            )
            unique_isos = np.setdiff1d(unique_isos, existing_scores.index)
            logger.info(
                f"Found {len(existing_scores)} existing scores. Remaining: {len(unique_isos)}"
            )
        except FileNotFoundError:
            logger.info("No existing scores found. Calculating all scores.")

    for start in range(0, len(unique_isos), 100):
        end = min(start + 100, len(unique_isos))

        logger.info(f"Calculating composite potential for isos {start:,} to {end:,}")

        data = {iso: calc_composite_potential(iso) for iso in unique_isos[start:end]}
        scores = pd.DataFrame.from_dict(
            data,
            orient="index",
            columns=[
                "agg_score_alex_norm",
                "agg_score_alex_deg",
                "agg_score_alex_det",
                "agg_score_edge_var",
                "agg_score_num_gen",
                "p_knotted",
                "count_unknotted",
                "all_knoted",
            ],
        )
        if start == 0 and restart:
            scores.to_csv(save_path / f"composite_scores.csv", index_label="iso")
        else:
            scores.to_csv(save_path / f"composite_scores.csv", mode="a", header=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # save_path = Path("mcmc") / "generic_samples"

    # isos_lists_df, path = run_mcmc(
    #     save_path,
    #     num_chains=7,
    #     gamma_=1 / 10,
    #     itts=10_000,
    #     steps=100,
    # )
    # logger.info(f"Saved MCMC samples to {path}")

    path = (
        Path("~")
        / "main"
        / "honours"
        / "pachner_graph_triangulations"
        / "data"
        / "results"
        / "mcmc"
        / "generic_samples"
        / "20251004_1845"
    )
    isos_lists_df = pd.read_csv(path / "isos_lists.csv")

    start_time = time.time()
    get_score(path, isos_lists_df, restart=False)
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time:.2f} seconds")
