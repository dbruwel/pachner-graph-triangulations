import multiprocessing

import numpy as np
import regina  # type: ignore

from .mcmc import neighbours


def calc_softmax(x, beta):
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / e_x.sum()


def take_step(iso, potential, beta):
    f_vector = regina.Triangulation3.fromIsoSig(iso).fVector()
    nbrs = neighbours(iso, f_vector, a=1)
    nbrs = list(nbrs.keys())
    if len(nbrs) == 0:
        p0 = potential(iso)
        return iso, p0[0], p0[0], p0[1], p0[2], p0[3]
    stats = np.array([potential(n) for n in nbrs])
    scores = [stat[0] for stat in stats]
    probs = calc_softmax(scores, beta)
    next_sample_idx = np.random.choice(np.arange(len(nbrs)), p=probs)

    next_sample = nbrs[next_sample_idx]
    score = scores[next_sample_idx]
    avg_score = np.mean(scores)

    p_knotted = stats[next_sample_idx][1]
    count_unknotted = stats[next_sample_idx][2]
    all_knoted = stats[next_sample_idx][3]

    return next_sample, score, avg_score, p_knotted, count_unknotted, all_knoted


def run_single_accent(base_iso, potential, beta, height=20):
    iso = base_iso
    isos = []
    scores = []
    avg_scores = []
    p_knotteds = []
    count_unknotteds = []
    all_knoteds = []

    for _ in range(height):
        iso, score, avg_score, p_knotted, count_unknotted, all_knoted = take_step(
            iso, potential, beta
        )
        isos.append(iso)
        scores.append(score)
        avg_scores.append(avg_score)
        p_knotteds.append(p_knotted)
        count_unknotteds.append(count_unknotted)
        all_knoteds.append(all_knoted)

    return isos, scores, avg_scores, p_knotteds, count_unknotteds, all_knoteds


def run_accent(base_isos, potential, beta, height=20):
    with multiprocessing.Pool() as pool:
        args = [
            (base_iso, potential, beta, height)
            for chain_id, base_iso in enumerate(base_isos)
        ]

        results = pool.starmap(run_single_accent, args)

    return results
