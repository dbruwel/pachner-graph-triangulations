# adapted from J. Spreer
import logging

from regina.engine import Triangulation3

logger = logging.getLogger(__name__)


def neighbours(iso: str, f: list[int], a: int) -> dict:
    """This function produces a dictionary of all `a`-neighbours (with isomorphism signatures as keys)
    of triangulation `iso` with f-vector `f`.
    This function uses dehydration signatures.

    Parameters
    ----------
    iso : str
        The dehydration signature of the triangulation.
    f : list[int]
        The f_vector of the triangulation.
    a : int
        The direction to move, a=1 is up, a=2 is down.

    Returns
    -------
    dict
        A dictionary of signature, move id pairs.
    """

    nbrs = {}
    # going up (2-3-moves at all triangles contained in two distinct tetrahedra)
    if a == 1:
        for t in range(f[2]):
            # create copy of tri in standard iso sig labelling
            target = Triangulation3.rehydrate(iso)
            # test if move is possible and if so, perform it
            if target.pachner(target.triangle(t), True, False):
                target.pachner(target.triangle(t), False, True)
                # get isomorphism signature of result, add it to neighbours
                try:
                    tiso = target.dehydrate()
                    # add edge needed to flip to obtain this neighbour (in standard iso sig labelling)
                    if tiso not in nbrs:
                        nbrs[tiso] = t
                except Exception:  # noqa
                    pass
        return nbrs
    # going down (3-2-move at every edge of degree three in three distinct tetrahedra)
    if a == 2:
        for e in range(f[1]):
            # create copy of tri in standard iso sig labelling
            target = Triangulation3.rehydrate(iso)
            # test if move is possible and if so, perform it
            if target.pachner(target.edge(e), True, False):
                target.pachner(target.edge(e), False, True)
                # get isomorphism signature of result, add it to neighbours
                try:
                    tiso = target.dehydrate()
                    # add edge needed to flip to obtain this neighbour (in standard iso sig labelling)
                    if tiso not in nbrs:
                        nbrs[tiso] = e
                except Exception:  # noqa
                    pass
        return nbrs

    else:
        return {}
