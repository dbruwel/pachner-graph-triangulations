import os
import time
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from pachner_traversal.potential_functions import Potential, VarianceEdgeDegree
from pachner_traversal.utils import data_root
from regina import Triangulation3


def compute_potential(iso: str) -> float | np.floating:
    iso_sig = Triangulation3.rehydrate(iso).isoSig()
    potential_val = Potential(VarianceEdgeDegree).calc_potential(iso_sig, 1)[0]
    return potential_val


def main():
    data_path = (
        data_root / "input_data" / "dehydration" / "processed" / "spheres_10.hdf5"
    )

    print("Loading data...")
    with h5py.File(data_path, "r") as f:
        data = f["isos"]
        isos = np.array(data[:])  # type: ignore
        isos = [iso.decode("utf-8") for iso in isos]

    isos_to_process = isos[:1_000]

    num_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    num_cores = min(100, num_cores)
    print(f"Spinning up {num_cores} workers...")

    tic = time.time()

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        ved_list = list(executor.map(compute_potential, isos_to_process))

    ved = np.array(ved_list)

    toc = time.time()
    time_taken = toc - tic

    print(f"Processed {len(isos_to_process)} items.")
    print(f"Time taken: {time_taken:,.2f} seconds")


if __name__ == "__main__":
    main()
