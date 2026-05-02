import os
import time
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from pachner_traversal.potential_functions import (
    Potential,
    VarianceEdgeDegree,
    DeterminantAlexanderPolynomial,
)
from pachner_traversal.utils import data_root
from regina import Triangulation3

import logging

logger = logging.getLogger(__name__)


def compute_potential_var(iso: str) -> float | np.floating:
    iso_sig = Triangulation3.rehydrate(iso).isoSig()
    potential_val = Potential(VarianceEdgeDegree).calc_potential(iso_sig, 1)[0]
    return potential_val


def compute_potential_det(iso: str) -> float | np.floating:
    iso_sig = Triangulation3.rehydrate(iso).isoSig()
    potential_val = Potential(DeterminantAlexanderPolynomial).calc_potential(
        iso_sig, 1
    )[0]
    return potential_val


def main():
    logging.basicConfig(level=logging.INFO)
    dataset_name = "edge_degree_variance"  # edge_degree_variance, det_alexander

    if dataset_name == "edge_degree_variance":
        compute_potential = compute_potential_var
    elif dataset_name == "det_alexander":
        compute_potential = compute_potential_det
    else:
        raise TypeError(f"invalid option {dataset_name}")

    data_path = (
        data_root / "input_data" / "dehydration" / "processed" / "spheres_10.hdf5"
    )

    logger.info("loading data")
    with h5py.File(data_path, "r") as f:
        data = f["isos"]
        isos = np.array(data[:])  # type: ignore
        isos = [iso.decode("utf-8") for iso in isos]

    isos_to_process = isos

    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    num_cores = min(100, num_cores)
    logger.info(f"number of workers: {num_cores}")

    tic = time.time()

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        ved_list = list(executor.map(compute_potential, isos_to_process))

    ved = np.array(ved_list)

    toc = time.time()
    time_taken = toc - tic

    logger.info(f"processed {len(isos_to_process)} items in {time_taken:,.2f} seconds.")

    logger.info("saving")

    with h5py.File(data_path, "r+") as f:
        if dataset_name in f:
            logger.info(f"overwriting")
            del f[dataset_name]

        f.create_dataset(dataset_name, data=ved, compression="gzip", compression_opts=4)


if __name__ == "__main__":
    main()
