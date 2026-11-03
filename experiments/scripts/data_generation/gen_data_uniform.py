"""Generates a uniform sample of a given manifiold via MCMC. Not efficient for large samples, not parallel."""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from pachner_traversal.classical_search.mcmc import sample_chain
from pachner_traversal.utils import get_data_root

logger = logging.getLogger(__name__)

data_root = get_data_root()

manifolds = {
    "s3a9": "cacabbwho",
    "s3b9": "cacabbwho",
    "s3c9": "cacabbwho",
    "s3d9": "cacabbwho",
    "s3e9": "cacabbwho",
    "s3f9": "cacabbwho",
    "s3g9": "cacabbwho",
    # "rp3": "cacabbwhg",
    # "l52": "dadbcccdxff",
    # "l72": "cacabbfhg",
    # "s2s1": "cacabbfxf",
    # "q8": "cabbbbhaq",
    # "e3": "gbpaacdefeffkfkloev",
    # "weeks": "jdlalaacfeghighiimhmxksrggb",
}


def run_manifold(manifold, seed, total_itts, out_dir):
    logger.info(f"Starting samples for {manifold}")

    results = sample_chain(
        seed=seed,
        gamma_=1 / 3.3,
        itts=total_itts,
        steps=50,
        chain_id=0,
    )
    with open(out_dir / f"{manifold}.txt", "w") as f:
        f.write("\n".join(results))
    return manifold


def main():
    samples = 100_000
    burnin = 5_000
    total_itts = samples + burnin

    out_dir = data_root / "input_data" / "dehydration" / "score_data" / "raw_data"
    assert out_dir.exists()
    with ProcessPoolExecutor(max_workers=7) as ex:
        futures = {
            ex.submit(run_manifold, manifold, seed, total_itts, out_dir): manifold
            for manifold, seed in manifolds.items()
        }
        for fut in as_completed(futures):
            logger.info(f"Finished {fut.result()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
