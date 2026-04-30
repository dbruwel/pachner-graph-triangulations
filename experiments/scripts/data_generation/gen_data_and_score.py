import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pachner_traversal.mcmc import run_chains
from pachner_traversal.potential_functions import calc_composite_potential
from pachner_traversal.utils import create_results_path, data_root

logger = logging.getLogger(__name__)


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
            existing_scores = pd.read_csv(save_path, index_col="iso")
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
            scores.to_csv(save_path, index_label="iso")
        else:
            scores.to_csv(save_path, mode="a", header=False)


# main function
def main():
    logging.basicConfig(level=logging.INFO)
    generate_data = False
    score_data = True

    if generate_data:
        save_path = create_results_path(Path("mcmc") / "generic_samples")
        num_chains = 7
        gamma_ = 1 / 10
        itts = 10_000
        steps = 100

        results = run_chains(
            num_chains=num_chains,
            seed="cMcabbgqs",
            gamma_=gamma_,
            itts=itts,
            steps=steps,
        )

        isos_lists_df = pd.DataFrame(results).T
        isos_lists_df.to_csv(save_path / "isos_lists.csv", index=False)
        logger.info(f"Saved MCMC samples to {save_path}")

    if score_data:
        save_path = data_root / "results" / "mcmc" / "generic_samples" / "20251004_1845"
        isos_lists_df = pd.read_csv(save_path / "isos_lists.csv")

        start_time = time.time()
        get_score(save_path / "composite_scores.csv", isos_lists_df, restart=False)
        end_time = time.time()

        logger.info(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
