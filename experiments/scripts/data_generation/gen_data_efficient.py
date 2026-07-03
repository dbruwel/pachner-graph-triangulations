import argparse
import collections
import logging
import random

from pachner_traversal.mcmc import iterate
from pachner_traversal.utils import get_data_root

data_root = get_data_root()

logger = logging.getLogger(__name__)

seeds = [
    "koddiacabcdefhhijjrjnjwijjiqb",
    "koddiacabcdefhhijjrjnjsljjhar",
    "koddiacabcdefihijjrvfjwdindvs",
    "koddiacabcdefiihjjrjnjsauusnk",
    "koddiacabcdffhhijjrvfjqtqqrtw",
    "koddiacabceefhhijjrjnootjjidc",
    "koddiacabceefhihjjrvfrfdjxwdc",
    "koddiacabcefeghijjrjgrxsrcpdc",
    "koddiacabcefehhijjrvfrhkrrqtw",
    "koddiacabcefehiijjrjnrxsvxuss",
    "koddiacabcefeihhjjrvfoxvxvwts",
    "koddiacabcefeihijjrvfrhblrmns",
    "koddiacabcefeihijjrvfwxfpfpcw",
    "koddiacabcefeihijjrvfgqwpjivo",
    "koddiacabcefeihijjrvbrxsxsmkf",
    "koddiacabceffghijjrngrxejsvmr",
    "koddiacabceffgihjjrjnrxejcrhr",
    "koddiacabceffhhijjrvfrqqqqfdr",
    "koddiacabceffhhijjrjnrpappges",
    "koddiacabceffhihjjrjsrxqhxwqk",
    "koddiacabceffhiijjrfbrxelxntv",
    "koddiacabceffhiijjrngrxeqxoiw",
    "koddiacabceffihhjjrvfrxqxdjej",
    "koddiacabceffiiijjrvfrxqtxnqc",
    "koddiacabcfeehhijjrvfxrnrrqtw",
    "koddiacabcfeehhijjrvftnwjjidc",
    "koddiacabcfeehiijjrjnagsnllsw",
    "koddiacabcfeeihijjrjndnwxsmnb",
    "koddiacabcfefgihjjrvfxrmrbjts",
    "koddiacabcfefhhijjrvfxrallodc",
    "koddiacabcfefhhijjrjnarmmmsqs",
    "koddiacabcfefhihjjrjnxrmhqjxk",
    "koddiacabcfefhiijjrvfxrmhxgiv",
    "koddiacabcfefhiijjrjnxrmmxrpk",
    "koddiacabcfefihhjjrfbxrmulchc",
    "koddiacabcfefihijjrjntrmxdkqv",
    "koddiacabcffegihjjrvfttwjkrqc",
    "koddiacabcffegihjjrjntxwjkrts",
    "koddiacabcffehhijjrvfxpsddgus",
    "koddiacaccdefghijjrwrvwtrwqqc",
    "koddiacaccdefgiijjrwrvwtrilrs",
    "koddiacaccdefgiijjrwrvwibaaoj",
    "koddiacaccdefhhijjrwrvwqjjhus",
    "koddiacaccdefhhijjrwfvwxvvaqb",
    "koddiacaccdefhihjjrwrvwqbxsic",
    "koddiacaccdefhihjjrwjvwekhfxn",
    "koddiacaccdefhihjjrwwvwevxcls",
    "koddiacaccdefhiijjrwrvwikxakj",
    "koddiacaccdefhiijjrwwvwefxujs",
    "koddiacaccdefhiijjrwfvwpnxlkj",
]


def generate_samples(leading_char, seed, gamma_, target_unique, steps):
    iso = seed
    discovered_isos = set()

    max_attempts = target_unique * 500
    attempts = 0

    while len(discovered_isos) < target_unique and attempts < max_attempts:
        iso = iterate(iso, gamma_, steps)

        if iso.startswith(leading_char):
            discovered_isos.add(iso)

        attempts += 1

    if attempts >= max_attempts:
        msg = f"Worker hit max attempts ({max_attempts}) before reaching target."
        logger.warning(msg)

    return list(discovered_isos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker_id", type=int, default=0, help="Unique ID for this parallel worker"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of outer loops to run"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    gamma_ = 1 / 5
    target_unique = 10_000
    steps = 1
    size = 10
    leading_char = "k"

    seed_buffer = collections.deque(maxlen=100)
    seed_buffer.extend(seeds)

    save_path = data_root / "input_data" / "dehydration" / "raw" / "s3_mcmc_samples_10"
    save_path.mkdir(parents=True, exist_ok=True)
    fname = f"samps_{size}_worker_{args.worker_id}.txt"

    with open(save_path / fname, "a") as f:
        for loop in range(args.iterations):
            logger.info(
                f"Worker {args.worker_id} starting batch {loop}/{args.iterations}"
            )
            seed = random.choice(tuple(seed_buffer))

            samples = generate_samples(
                leading_char=leading_char,
                seed=seed,
                gamma_=gamma_,
                target_unique=target_unique,
                steps=steps,
            )

            if samples:
                seed_buffer.extend(samples)
                f.writelines(f"{s}\n" for s in samples)
                f.flush()

    logger.info(f"Worker {args.worker_id} finished generation.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
