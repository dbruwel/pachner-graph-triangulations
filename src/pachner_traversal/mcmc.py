import math
import random

from regina.engine import Triangulation3
import logging
import multiprocessing
from datetime import datetime

logger = logging.getLogger(__name__)


def neighbours(iso: str, f: list[int], a: int) -> dict:
    "This function produces a dictionary of all `a`-neighbours (with isomorphism signatures as keys) of triangulation `iso` with f-vector `f`. This function uses non-standard isomorphism signatures and hence requires `regina` version 7.3 or newer."

    nbrs = {}
    # going up (2-3-moves at all triangles contained in two distinct tetrahedra)
    if a == 1:
        for t in range(f[2]):
            # create copy of tri in standard iso sig labelling
            target = Triangulation3.fromIsoSig(iso)
            # test if move is possible and if so, perform it
            if target.pachner(target.triangle(t), True, False):
                target.pachner(target.triangle(t), False, True)
                # get isomorphism signature of result, add it to neighbours
                tiso = target.isoSig_RidgeDegrees()
                # add edge needed to flip to obtain this neighbour (in standard iso sig labelling)
                if not tiso in nbrs:
                    nbrs[tiso] = t
        return nbrs
    # going down (3-2-move at every edge of degree three in three distinct tetrahedra)
    if a == 2:
        for e in range(f[1]):
            # create copy of tri in standard iso sig labelling
            target = Triangulation3.fromIsoSig(iso)
            # test if move is possible and if so, perform it
            if target.pachner(target.edge(e), True, False):
                target.pachner(target.edge(e), False, True)
                # get isomorphism signature of result, add it to neighbours
                tiso = target.isoSig_RidgeDegrees()
                # add edge needed to flip to obtain this neighbour (in standard iso sig labelling)
                if not tiso in nbrs:
                    nbrs[tiso] = e
        return nbrs

    else:
        return {}


def choosemove(iso: str, f: list[int], gamma: float) -> tuple[str, list[int]]:
    "This function takes a state triangulation given by isomorphism signature `iso` with f-vector `f` and paramter `gamma`. It computes a proposal, enumerates neighbours of `iso` and decides wether to perform the proposed move."
    x = random.random()
    if x < math.exp((-1) * gamma * f[3]):
        a = 1
    else:
        a = 2
    ngbrs = neighbours(iso, f, a)
    num_ngbrs = len(ngbrs.keys())
    # random number for proposal to move or stay
    i = random.random()
    # setup done

    # go up (2-3)
    if a == 1:
        # stay where you are
        if i > float(num_ngbrs) / float(f[2]):
            return iso, f
        # move up (very likely)
        else:
            return random.choice(list(ngbrs.keys())), [
                f[0],
                f[1] + 1,
                f[2] + 2,
                f[3] + 1,
            ]
    # go down
    elif a == 2:
        # stay where you are
        if f[3] <= 2:
            return iso, f
        if i > float(num_ngbrs) / float(f[2] - 2):
            return iso, f
        # go down (unlikely)
        else:
            return random.choice(list(ngbrs.keys())), [
                f[0],
                f[1] - 1,
                f[2] - 2,
                f[3] - 1,
            ]
    return "", []


def randomise(
    iso: str,
    f: list[int],
    steps: int,
    gamma: float,
    interval: float,
    offset: float,
    name: str,
) -> bool:
    "This is the main function taking in see triangulation `iso` with f-vector `f`. It performs a random walk in the Pachner graph of length `steps` with parameter `gamma`. Parameter `verbose` decides print behaviour, `name` is the filename for the output file."
    # initialise number of steps
    st = 0
    with open(name, "w") as fl:
        fl.write("")
    while st < steps + offset * interval - 1:
        st += 1
        iso, f = choosemove(iso, f, gamma)
        if interval != 0 and st % interval == 0 and st >= offset * interval:
            # open output file
            with open(name, "a") as fl:
                fl.write(iso + "\n")
            print(
                "collecting triangulation",
                int((st - offset * interval) / interval + 1),
                ":",
                iso,
            )
    return True


def iterate(iso: str, gamma: float, steps: int = 1) -> str:
    # initialise number of steps
    t = Triangulation3.fromIsoSig(iso)
    f = t.fVector()
    samp = 0

    for i in range(int(steps)):
        iso, f = choosemove(iso, f, gamma)
    return iso


def mcmc3d(
    iso, gamma, samples=10, offset=0, interval=100, verbose=True, printToFile=False
):
    "Collects 'samples' samples of triangulations by performing a random walk in the Pachner graph starting from 'iso' with parameter `gamma`. offset' is the number of triangulations to be burnt (discarded initially). 'interval' is the number of triangulations between successive samples. Parameter `verbose` decides print behaviour, `printToFile` is the filename for the output file in the folder outputs/."
    # initialise number of steps
    t = Triangulation3.fromIsoSig(iso)
    f = t.fVector()
    samp = 0

    # output
    name = (
        "outputs/mcmc3d_"
        + str(printToFile)
        + "_gamma_"
        + str(gamma)
        + "_samples_"
        + str(samples)
        + ".txt"
    )

    # burn
    if offset > 0:
        for i in range(int(offset)):
            iso, f = choosemove(iso, f, gamma)

    while samp < samples:
        # interval between samples
        for i in range(int(interval)):
            iso, f = choosemove(iso, f, gamma)

        # sample
        samp += 1
        if verbose:
            print(
                "collecting triangulation", int(samp), " of ", int(samples), " :", iso
            )
        if printToFile != False:
            # open output file
            with open(name, "a") as fl:
                fl.write(iso + "\n")
    return True


###


def sample_chain(
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
    chain_id: int,
):
    isos = [seed]

    for itt in range(itts):
        if (itt % 1_000 == 0) or (itt < 100 and itt % 10 == 0):
            if chain_id == 0:
                logger.info(
                    f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f} at {datetime.now().strftime('%H:%M:%S')}"
                )
        current_iso = isos[-1]

        proposed_iso = iterate(current_iso, gamma_, steps=steps)
        isos.append(proposed_iso)

    return isos


def run_chains(
    num_chains: int,
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
) -> list[list[str]]:
    logger.info(f"Running {num_chains:,.0f} chains with {itts:,.0f} iterations each.")

    with multiprocessing.Pool(processes=num_chains) as pool:
        args = [
            (
                seed,
                gamma_,
                itts,
                steps,
                chain_id,
            )
            for chain_id in range(num_chains)
        ]

        isos_lists = pool.starmap(sample_chain, args)

    return isos_lists
