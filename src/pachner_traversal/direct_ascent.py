import functools
import logging
import multiprocessing
from multiprocessing import Pool

import numpy as np
import regina

from .mcmc import neighbours

logger = logging.getLogger(__name__)


def calc_softmax(x, beta):
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / e_x.sum()


def worker_function(n, potential):
    return potential(n)[0]


@functools.lru_cache(maxsize=1)
def take_step(iso, potential, beta):
    f_vector = regina.engine.Triangulation3.fromIsoSig(iso).fVector()
    nbrs = neighbours(iso, f_vector, a=1)
    nbrs = list(nbrs.keys())
    logger.debug(f"Found {len(nbrs)} neighbours for {iso}.")
    if len(nbrs) == 0:
        p0 = potential(iso)
        return iso, p0[0], p0[0], p0[1], p0[2], p0[3]

    with Pool(processes=7) as pool:
        results_list = pool.starmap(worker_function, [(n, potential) for n in nbrs])
    scores = np.array(results_list)
    if beta == np.inf:
        logger.debug("Using greedy selection.")
        next_sample_idx = np.argmax(scores)
    else:
        probs = calc_softmax(scores, beta)
        next_sample_idx = np.random.choice(np.arange(len(nbrs)), p=probs)

    next_sample = nbrs[next_sample_idx]
    with open("debug.txt", "a") as f:
        f.write(f"{next_sample}\n")
    score = scores[next_sample_idx]
    avg_score = np.mean(scores)

    return next_sample, score, avg_score, None, None, None


def run_single_accent(chain_id, base_iso, potential, beta, height=20):
    iso = base_iso
    isos = []
    scores = []
    avg_scores = []
    p_knotteds = []
    count_unknotteds = []
    all_knoteds = []

    logger.info(f"Running chain {chain_id}, starting at {base_iso}.")

    for h in range(height):
        logger.debug(f"Current iso: {iso} at height {h}.")
        iso, score, avg_score, p_knotted, count_unknotted, all_knoted = take_step(
            iso, potential, beta
        )
        isos.append(iso)
        scores.append(score)
        avg_scores.append(avg_score)
        p_knotteds.append(p_knotted)
        count_unknotteds.append(count_unknotted)
        all_knoteds.append(all_knoted)

    return isos, scores, avg_scores, p_knotteds, count_unknotteds, all_knoteds, beta


def run_accent(base_isos, potential, betas, height=20):
    logger.info(f"Running {len(base_isos)} chains of height {height}.")
    # with Pool() as pool:
    #     args = [
    #         (chain_id, base_iso, potential, beta, height)
    #         for chain_id, (base_iso, beta) in enumerate(zip(base_isos, betas))
    #     ]

    #     results = pool.starmap(run_single_accent, args)

    args = [
        (chain_id, base_iso, potential, beta, height)
        for chain_id, (base_iso, beta) in enumerate(zip(base_isos, betas))
    ]

    results = [run_single_accent(*arg) for arg in args]

    return results
