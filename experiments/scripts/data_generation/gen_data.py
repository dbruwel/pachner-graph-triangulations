import logging
from datetime import datetime

import numpy as np

from pachner_traversal.mcmc import run_chains
from pachner_traversal.utils import data_root, leading_chars

logger = logging.getLogger(__name__)


# main function
def main():
    logging.basicConfig(level=logging.INFO)
    num_chains = 30
    seed = "cMcabbgqs"
    gamma_ = 1 / 10
    itts = 1_000_000
    steps = 1
    size = 20
    leading_char = leading_chars[size]

    save_path = data_root / "input_data" / "dehydration" / "raw" / "mcmc_samples"

    for i in range(10):
        logger.info(
            f"Starting MCMC run {i + 1}/10 at {datetime.now().strftime('%H:%M:%S')}"
        )

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

        logger.info(f"Total unique samples: {len(isos_list):,}.")
        logger.info(f"Example samples: {isos_list[-5:]}")

        samp = isos_list[np.char.startswith(isos_list, leading_char)]

        logger.info(f"{len(samp):,} samples for N={size}.")
        with open(save_path / f"samps{size}.txt", "a") as f:
            np.savetxt(f, samp, fmt="%s")

        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "mcmc_samples.txt", "a") as f:
            np.savetxt(f, isos_list, fmt="%s")


if __name__ == "__main__":
    main()
