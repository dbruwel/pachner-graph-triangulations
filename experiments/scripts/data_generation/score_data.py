import logging
import multiprocessing

import h5py
import numpy as np
from pachner_traversal.utils import get_data_root, logger_config
from regina import Triangulation3
from snappy import Manifold

logger = logging.getLogger(__name__)

data_root = get_data_root()


def compute_potential_degree_composite(iso: str) -> dict:
    tri = Triangulation3.rehydrate(iso)
    degs = []

    for edge in tri.edges():
        degs.append(edge.degree())

    degs = np.array(degs)

    res = {}

    res["edge_degree_variance"] = np.var(degs)
    res["count_1_deg"] = (degs == 1).sum()
    res["count_2_deg"] = (degs == 2).sum()
    res["count_3_deg"] = (degs == 3).sum()
    res["count_4_deg"] = (degs == 4).sum()
    res["count_5_deg"] = (degs == 5).sum()

    return res


def compute_potential_var(iso: str) -> float | np.floating:
    tri = Triangulation3.rehydrate(iso)
    degs = []

    for edge in tri.edges():
        degs.append(edge.degree())

    potential_value = np.var(degs)
    return potential_value


def compute_potential_det(iso: str) -> float | np.floating:
    tri = Triangulation3.rehydrate(iso)
    n_edges = len(tri.edges())
    scores = []

    for i_edge in range(n_edges):
        temp_tri = Triangulation3.rehydrate(iso)
        edge = temp_tri.edge(i_edge)
        temp_tri.pinchEdge(edge)
        m = Manifold(temp_tri)
        alex_poly = m.alexander_polynomial()
        score = np.abs(alex_poly(-1))
        scores.append(score)

    potential_value = np.mean(scores)
    return potential_value


def compute_potential_det_safe(iso: str) -> float | np.floating:
    result = None
    for _ in range(10):
        pool = multiprocessing.Pool(processes=1)
        async_result = pool.apply_async(compute_potential_det, (iso,))
        try:
            result = async_result.get(timeout=3)

            pool.close()
            pool.join()
            break

        except multiprocessing.TimeoutError:
            pool.terminate()
            pool.join()

    if result is None:
        raise RuntimeError(f"`{iso}` failed to run.")

    return result


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


potential_map = {
    # "det_alexander": compute_potential_det_safe,
    "loop_count": compute_potential_loops,
}


def main(raw_data_path, save_path):
    logging.basicConfig(**logger_config)

    logger.info("loading data")
    with open(raw_data_path, "r") as f:
        sigs = f.readlines()
        sigs = [sig.strip() for sig in sigs]

    max_len = len(sigs[0])

    scores = {}

    for dataset_name in potential_map:
        logger.info(f"Scoring `{dataset_name}`")
        scores[dataset_name] = []
        for i, sig in enumerate(sigs):
            if i % 1_000 == 0:
                logger.info(f"Processing signature {i:,} / {len(sigs):,}")

            result = potential_map[dataset_name](sig)
            scores[dataset_name].append(result)

    data = [compute_potential_degree_composite(sig) for sig in sigs]
    scores_degree = {k: [d[k] for d in data] for k in data[0]}

    with h5py.File(save_path, "w") as f:
        dt = h5py.string_dtype(encoding="utf-8", length=max_len)
        f.create_dataset("isos", data=sigs, dtype=dt)
        for dataset_name in scores:
            f.create_dataset(dataset_name, data=scores[dataset_name])
        for dataset_name in scores_degree:
            f.create_dataset(dataset_name, data=scores_degree[dataset_name])


if __name__ == "__main__":
    raw_data_path = "/home/dbruwel/main/honours/pachner_graph_triangulations/data/input_data/dehydration/raw/spheres_10_4m.txt"
    save_path = "/home/dbruwel/main/honours/pachner_graph_triangulations/data/input_data/dehydration/processed/spheres_10_scored.hdf5"
    main(raw_data_path, save_path)
