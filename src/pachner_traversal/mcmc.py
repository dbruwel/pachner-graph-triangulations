import math
import random

from regina import *  # type: ignore


def neighbours(iso, f, a):
    "This function produces a dictionary of all `a`-neighbours (with isomorphism signatures as keys) of triangulation `iso` with f-vector `f`. This function uses non-standard isomorphism signatures and hence requires `regina` version 7.3 or newer."

    nbrs = {}
    # going up (2-3-moves at all triangles contained in two distinct tetrahedra)
    if a == 1:
        for t in range(f[2]):
            # create copy of tri in standard iso sig labelling
            target = Triangulation3.fromIsoSig(iso)  # type: ignore
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
            target = Triangulation3.fromIsoSig(iso)  # type: ignore
            # test if move is possible and if so, perform it
            if target.pachner(target.edge(e), True, False):
                target.pachner(target.edge(e), False, True)
                # get isomorphism signature of result, add it to neighbours
                tiso = target.isoSig_RidgeDegrees()
                # add edge needed to flip to obtain this neighbour (in standard iso sig labelling)
                if not tiso in nbrs:
                    nbrs[tiso] = e
        return nbrs


def choosemove(iso, f, gamma):
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
    return None


def randomise(iso, f, steps, gamma, interval, offset, name):
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


def iterate(iso, gamma, steps=1):
    # initialise number of steps
    t = Triangulation3.fromIsoSig(iso)  # type: ignore
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
    t = Triangulation3.fromIsoSig(iso)  # type: ignore
    f = t.fVector()
    samp = 0

    # output
    if printToFile != False:
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
