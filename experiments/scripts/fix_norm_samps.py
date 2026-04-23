import numpy as np
from pachner_traversal.utils import data_path
from regina import Triangulation3


def normalise(N):
    new_read_path = (
        data_path
        / "input_data"
        / "dehydration"
        / "raw"
        / "mcmc_samples"
        / f"samps{N}.txt"
    )
    old_read_path = (
        data_path / "input_data" / "dehydration" / "raw" / f"d_training_spheres_{N}.txt"
    )

    out_path = (
        data_path
        / "input_data"
        / "dehydration"
        / "raw"
        / "norm_samps"
        / f"samps{N}.txt"
    )

    new_arr = np.loadtxt(new_read_path, dtype=str)
    new_arr = np.unique(new_arr)
    new_arr = new_arr[:1_000_000]

    res = []

    for iso in new_arr:
        tri = Triangulation3.fromIsoSig(iso)
        res.append(tri.dehydrate())

    res = np.array(res)
    if len(res) < 1_000_000:
        old_arr = np.loadtxt(old_read_path, dtype=str)
        arr = np.concatenate((res, old_arr))
        arr = np.unique(arr)
        arr = arr[:1_000_000]
    else:
        arr = res

    if len(arr) < 1_000_000:
        print(f"Only {len(arr):,} unique samples for N={N}.")

    with open(out_path, "w") as f:
        np.savetxt(f, arr, fmt="%s")


if __name__ == "__main__":
    for N in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        normalise(N)
