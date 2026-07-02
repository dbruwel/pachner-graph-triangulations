import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from pachner_traversal.potential_functions import (
    DeterminantAlexanderPolynomial,
    Potential,
    VarianceEdgeDegree,
)
from pachner_traversal.utils import get_data_root
from regina import Triangulation3

logger = logging.getLogger(__name__)

data_root = get_data_root()


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


def compute_potential_loops(iso: str):
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0

    for tet in tri.tetrahedra():
        for i in range(4):
            loop = tet.index() == tet.adjacentSimplex(i).index()
            potential_val = potential_val + loop / 2

    return potential_val


def compute_potential_unit_degree(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        unit_degree = float(edge.degree() == 1)
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_1_degree(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        unit_degree = float(edge.degree() == 1)
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_2_degree(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        unit_degree = float(edge.degree() == 2)
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_3_degree(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        unit_degree = float(edge.degree() == 3)
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_3_degree_regular(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        if edge.degree() == 3:
            if tri.pachner(edge, True, False):
                unit_degree = 1.0
            else:
                unit_degree = 0.0
        else:
            unit_degree = 0.0
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_3_degree_folded(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        if edge.degree() == 3:
            if tri.pachner(edge, True, False):
                unit_degree = 0.0
            else:
                unit_degree = 1.0
        else:
            unit_degree = 0.0
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_4_degree(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        unit_degree = float(edge.degree() == 4)
        potential_val = potential_val + unit_degree

    return potential_val


def compute_potential_5_degree(iso: str) -> float:
    tri = Triangulation3.rehydrate(iso)
    potential_val = 0.0

    for edge in tri.edges():
        unit_degree = float(edge.degree() == 5)
        potential_val = potential_val + unit_degree

    return potential_val


def score_data(dataset_name):
    if dataset_name == "edge_degree_variance":
        compute_potential = compute_potential_var
    elif dataset_name == "det_alexander":
        compute_potential = compute_potential_det
    elif dataset_name == "loop_count":
        compute_potential = compute_potential_loops
    elif dataset_name == "unit_deg":
        compute_potential = compute_potential_unit_degree
    elif dataset_name == "count_1_deg":
        compute_potential = compute_potential_1_degree
    elif dataset_name == "count_2_deg":
        compute_potential = compute_potential_2_degree
    elif dataset_name == "count_3_deg":
        compute_potential = compute_potential_3_degree
    elif dataset_name == "count_4_deg":
        compute_potential = compute_potential_4_degree
    elif dataset_name == "count_5_deg":
        compute_potential = compute_potential_5_degree
    elif dataset_name == "count_3_deg_regular":
        compute_potential = compute_potential_3_degree_regular
    elif dataset_name == "count_3_deg_folded":
        compute_potential = compute_potential_3_degree_folded
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

    if num_cores == 1:
        ved_list = [compute_potential(iso) for iso in isos_to_process]
    else:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            ved_list = list(executor.map(compute_potential, isos_to_process))

    ved = np.array(ved_list)

    toc = time.time()
    time_taken = toc - tic

    logger.info(f"processed {len(isos_to_process)} items in {time_taken:,.2f} seconds.")

    logger.info("saving")

    with h5py.File(data_path, "r+") as f:
        if dataset_name in f:
            logger.info("overwriting")
            del f[dataset_name]

        f.create_dataset(dataset_name, data=ved, compression="gzip", compression_opts=4)


def main():
    logging.basicConfig(level=logging.INFO)
    score_data("count_3_deg_regular")
    score_data("count_3_deg_folded")
    # score_data("count_2_deg")
    # score_data("count_3_deg")
    # score_data("count_4_deg")
    # score_data("count_5_deg")


if __name__ == "__main__":
    main()
