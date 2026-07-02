import logging
import random

import numpy as np
from pachner_traversal.data import leading_chars
from pachner_traversal.mcmc import run_chain_multi_arg
from pachner_traversal.utils import get_data_root

logger = logging.getLogger(__name__)


def generate_samples(
    leading_char,
    num_chains,
    seeds,
    gamma_,
    itts,
    steps,
):
    # Runs the MCMC chains.
    data = run_chain_multi_arg(
        num_chains=num_chains,
        itts=itts,
        seed_list=seeds,
        gamma_list=[gamma_] * num_chains,
        steps_list=[steps] * num_chains,
    )

    isos_list_target = [
        str(x) for x in np.array(data).flatten() if str(x).startswith(leading_char)
    ]

    return isos_list_target


# Main function.
def main():
    # Setup.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    noisy_logger = logging.getLogger("pachner_traversal.mcmc")
    noisy_logger.setLevel(logging.CRITICAL)

    num_chains = 50
    seed = "cMcabbgqs"
    gamma_ = 1 / 6
    itts = 100_000
    steps = 1
    size = 15
    leading_char = leading_chars[size]

    # Data containers - we maintain a list and set to avoid continuously converting
    # the list to a set for membership checks, which is costly as the list grows.
    unique_set = set()
    unique_list = []

    # Pathing.
    data_root = get_data_root()
    save_path = data_root / "input_data" / "dehydration" / "raw" / "mcmc_samples"
    fname = f"samps{size}_500m.txt"

    # Loads existing samples into memory.
    if (save_path / fname).exists():
        logger.info(f"File {fname} already exists. Loading existing samples.")
        with open(save_path / fname, "r") as f:
            for line in f:
                item = line.strip()
                if item:
                    unique_set.add(item)
                    unique_list.append(item)

    # Opens file once in append mode.
    with open(save_path / fname, "a") as f:
        while len(unique_set) <= 500_000_000:
            # Randomly samples seeds from existing samples. This reduces warmup time.
            if len(unique_list) > num_chains:
                seeds = random.sample(unique_list, num_chains)
            else:
                seeds = [seed] * num_chains

            # Generate samples and filter by leading char.
            isos_list_target = generate_samples(
                leading_char=leading_char,
                num_chains=num_chains,
                seeds=seeds,
                gamma_=gamma_,
                itts=itts,
                steps=steps,
            )

            # Use set difference to find new samples - very efficient.
            new_samples = list(set(isos_list_target) - unique_set)

            if new_samples:
                # Update our data containers and write new samples to file.
                unique_set.update(new_samples)
                unique_list.extend(new_samples)

                f.writelines(f"{s}\n" for s in new_samples)
                f.flush()

            logger.info(f"{len(unique_set):,} samples for N={size}.")


if __name__ == "__main__":
    main()
