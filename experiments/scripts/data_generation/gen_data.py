import logging
from datetime import datetime

import numpy as np
from pachner_traversal.mcmc import run_chains
from pachner_traversal.utils import data_root, leading_chars

logger = logging.getLogger(__name__)


# main function
def main():
    logging.basicConfig(level=logging.INFO)
    num_chains = 80
    seed = "cMcabbgqs"
    gamma_ = 1 / 10
    itts = 1_000_000
    steps = 1
    size = 15
    leading_char = leading_chars[size]

    unique_target_samps = set()

    save_path = data_root / "input_data" / "dehydration" / "raw" / "mcmc_samples"

    while len(unique_target_samps) <= 10_005_000:
        logger.info(f"Starting MCMC run at {datetime.now().strftime('%H:%M:%S')}")

        data = run_chains(
            num_chains=num_chains,
            seed=seed,
            gamma_=gamma_,
            itts=itts,
            steps=steps,
        )

        isos_list = np.array(data).flatten()
        isos_list = isos_list.astype(str)
        isos_list = np.unique(isos_list)

        isos_list_target = isos_list[np.char.startswith(isos_list, leading_char)]
        unique_target_samps.update(isos_list_target)

    logger.info(f"{len(unique_target_samps):,} samples for N={size}.")
    with open(save_path / f"samps{size}.txt", "a") as f:
        np.savetxt(f, list(unique_target_samps), fmt="%s")


if __name__ == "__main__":
    main()
