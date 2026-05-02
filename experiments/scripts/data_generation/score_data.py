import time

import h5py
import numpy as np
from pachner_traversal.potential_functions import Potential, VarianceEdgeDegree
from pachner_traversal.utils import data_root
from regina import Triangulation3


def main():
    data_path = (
        data_root / "input_data" / "dehydration" / "processed" / "spheres_10.hdf5"
    )

    with h5py.File(data_path, "r") as f:
        data = f["isos"]
        isos = np.array(data[:])  # type: ignore
        isos = np.array([iso.decode("utf-8") for iso in isos])

    print(isos)
    tic = time.time()
    ved = [
        Potential(VarianceEdgeDegree).calc_potential(
            Triangulation3.rehydrate(iso).isoSig(), 1
        )[0]
        for iso in isos[:100]
    ]
    ved = np.array(ved)
    toc = time.time()
    time_taken = toc - tic

    print(f"Time taken: {time_taken:,.2f}")


if __name__ == "__main__":
    main()
