# adapted from J. Spreer
"""Faster variant of `choosemove` from `functions3d.py` (Algorithm 2 of Altmann & Spreer).

The original `choosemove` enumerates every distinct neighbour (up to isomorphism)
of the current triangulation at each step, which requires computing isomorphism
signatures for every candidate move. This version instead picks a
single valid site (triangle or edge) uniformly at random, performs the move, and
corrects for the resulting bias with a Metropolis-Hastings step using the ratio
of automorphism group sizes. Only one signature is computed per step.

The up/down proposal is the same as the original: an up move (2-3) is proposed
with probability exp(-gamma * f_3), otherwise a down move (3-2).
"""

import logging
import math
import multiprocessing
import random
from datetime import datetime

from regina.engine import Triangulation3

logger = logging.getLogger(__name__)


def get_aut(tri: Triangulation3) -> int:
    """Return the size of the automorphism group of a triangulation.

    Parameters
    ----------
    tri : Triangulation3
        The triangulation.

    Returns
    -------
    int
        The number of combinatorial automorphisms of `tri`.
    """
    return len(tri.findAllIsomorphisms(tri))


def get_f_vector(tri: Triangulation3) -> list[int]:
    """Return the f-vector of a triangulation.

    Parameters
    ----------
    tri : Triangulation3
        The triangulation.

    Returns
    -------
    list[int]
        The f-vector (f_0, f_1, f_2, f_3).
    """
    f = tri.fVector()
    return f


def perform_move(tri: Triangulation3, a: int, i: int) -> Triangulation3:
    """Perform a Pachner move at a given site, without checking validity.

    The input triangulation is copied and left unchanged.

    Parameters
    ----------
    tri : Triangulation3
        The initial triangulation.
    a : int
        The move type. 1 is a 2-3 move (up, on triangle `i`), 2 is a 3-2 move (down, on edge `i`).
    i : int
        The index of the triangle (a=1) or edge (a=2) to move on. Should come from `get_sites`.

    Returns
    -------
    Triangulation3
        The modified copy of `tri`.
    """
    target = Triangulation3(tri)

    if a == 1:
        target.pachner(target.triangle(i), False, True)
    if a == 2:
        target.pachner(target.edge(i), False, True)

    return target


def get_sites(tri: Triangulation3, f: list[int], a: int) -> list[int]:
    """List all sites at which a given move type can be performed.

    Parameters
    ----------
    tri : Triangulation3
        The triangulation.
    f : list[int]
        The f-vector of `tri`.
    a : int
        The move type, see `perform_move`.

    Returns
    -------
    list[int]
        Indices of the triangles (a=1) or edges (a=2) at which the move is valid.
    """
    sites = []
    target = Triangulation3(tri)

    if a == 1:
        for t in range(f[2]):
            if target.pachner(target.triangle(t), True, False):
                sites.append(t)

    if a == 2:
        for e in range(f[1]):
            if target.pachner(target.edge(e), True, False):
                sites.append(e)

    return sites


### Choose move from proposal
def choosemove(
    sig: str,
    gamma: float,
    f: list[int] | None = None,
    aut: int | None = None,
) -> tuple[str, list[int], int]:
    """Perform one step of the MCMC chain.

    Proposes an up move with probability exp(-gamma * f_3) (else a down move),
    stays put with the same probability as the original `choosemove`, and
    otherwise moves at a uniformly random valid site. The move is accepted with
    probability min(1, |Aut(T')| / |Aut(T)|).

    Passing `f` and `aut` from the previous step avoids recomputing them.

    Parameters
    ----------
    sig : str
        Dehydration signature of the current triangulation.
    gamma : float
        Parameter controlling the up/down proposal, and hence the typical number of tetrahedra.
    f : list[int] | None, optional
        f-vector of the current triangulation. Computed if None.
    aut : int | None, optional
        Automorphism group size of the current triangulation. Computed if None.

    Returns
    -------
    tuple[str, list[int], int]
        The dehydration signature, f-vector and automorphism group size of the next state.

    Raises
    ------
    RuntimeError
        If no branch returns (should never happen).
    """
    tri = Triangulation3.rehydrate(sig)

    if aut is None:
        aut = get_aut(tri)
    if f is None:
        f = get_f_vector(tri)

    x = random.random()
    if x < math.exp((-1) * gamma * f[3]):
        a = 1
    else:
        a = 2

    sites = get_sites(tri, f, a)
    num_sites = len(sites)
    i = random.random()
    u = random.random()

    # go up (2-3)
    if a == 1:
        if i > float(num_sites) / float(f[2]):
            return tri.dehydrate(), f, aut
        else:
            tri_prop = perform_move(tri, a, random.choice(sites))
            aut_prop = get_aut(tri_prop)
            accept_prob = min(1, aut_prop / aut)
            if u > accept_prob:
                return tri.dehydrate(), f, aut
            return (
                tri_prop.dehydrate(),
                [
                    f[0],
                    f[1] + 1,
                    f[2] + 2,
                    f[3] + 1,
                ],
                aut_prop,
            )

    # go down
    elif a == 2:
        if f[3] <= 2:
            return tri.dehydrate(), f, aut
        if i > float(num_sites) / float(f[2] - 2):
            return tri.dehydrate(), f, aut
        else:
            tri_prop = perform_move(tri, a, random.choice(sites))
            aut_prop = get_aut(tri_prop)
            accept_prob = min(1, aut_prop / aut)
            if u > accept_prob:
                return tri.dehydrate(), f, aut
            return (
                tri_prop.dehydrate(),
                [
                    f[0],
                    f[1] - 1,
                    f[2] - 2,
                    f[3] - 1,
                ],
                aut_prop,
            )

    raise RuntimeError("Should not reach this point in choosemove")


def randomise(
    sig: str,
    steps: int,
    gamma: float,
    interval: float,
    offset: float,
    name: str,
    f: list[int] | None = None,
    aut: int | None = None,
) -> bool:
    "This is the main function taking in see triangulation `sig` with f-vector `f`. It performs a random walk in the Pachner graph of length `steps` with parameter `gamma`. Parameter `verbose` decides print behaviour, `name` is the filename for the output file."
    # initialise number of steps
    st = 0
    with open(name, "w") as fl:
        fl.write("")
    while st < steps + offset * interval - 1:
        st += 1
        sig, f, aut = choosemove(sig, f=f, gamma=gamma, aut=aut)
        if interval != 0 and st % interval == 0 and st >= offset * interval:
            # open output file
            with open(name, "a") as fl:
                fl.write(sig + "\n")
            print(
                "collecting triangulation",
                int((st - offset * interval) / interval + 1),
                ":",
                sig,
            )
    return True


def iterate(sig: str, gamma: float, steps: int = 1) -> str:
    # initialise number of steps
    t = Triangulation3.rehydrate(sig)
    f = t.fVector()
    aut = get_aut(t)

    for _ in range(int(steps)):
        sig, f, aut = choosemove(sig, f=f, gamma=gamma, aut=aut)
    return sig


def mcmc3d(
    sig, gamma, samples=10, offset=0, interval=100, verbose=True, printToFile=False
):
    "Collects 'samples' samples of triangulations by performing a random walk in the Pachner graph starting from 'sig' with parameter `gamma`. offset' is the number of triangulations to be burnt (discarded initially). 'interval' is the number of triangulations between successive samples. Parameter `verbose` decides print behaviour, `printToFile` is the filename for the output file in the folder outputs/."
    # initialise number of steps
    t = Triangulation3.rehydrate(sig)
    f = t.fVector()
    aut = get_aut(t)
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
            sig, f, aut = choosemove(sig, f=f, gamma=gamma, aut=aut)

    while samp < samples:
        # interval between samples
        for i in range(int(interval)):
            sig, f, aut = choosemove(sig, f=f, gamma=gamma, aut=aut)

        # sample
        samp += 1
        if verbose:
            print(
                "collecting triangulation", int(samp), " of ", int(samples), " :", sig
            )
        if printToFile:
            # open output file
            with open(name, "a") as fl:
                fl.write(sig + "\n")
    return True


###


def sample_chain(
    seed: str,
    gamma_: float,
    itts: int,
    steps: int,
    chain_id: int,
):
    sigs = [seed]

    for itt in range(itts):
        if (itt % 10_000 == 0) and (chain_id == 0):
            logger.info(
                f"Chain {chain_id}: iteration {itt:,.0f}/{itts:,.0f} at {datetime.now().strftime('%H:%M:%S')}"
            )
        current_sig = sigs[-1]

        proposed_sig = iterate(current_sig, gamma_, steps=steps)
        sigs.append(proposed_sig)

    return sigs


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

        sigs_lists = pool.starmap(sample_chain, args)

    return sigs_lists


def run_chain_multi_arg(
    num_chains: int,
    itts: int,
    seed_list: list[str],
    gamma_list: list[float],
    steps_list: list[int],
) -> list[list[str]]:
    def get_args(chain_id: int):
        args = (
            seed_list[chain_id],
            gamma_list[chain_id],
            itts,
            steps_list[chain_id],
            chain_id,
        )
        return args

    with multiprocessing.Pool(processes=num_chains) as pool:
        args = [get_args(chain_id) for chain_id in range(num_chains)]

        sigs_lists = pool.starmap(sample_chain, args)

    return sigs_lists
