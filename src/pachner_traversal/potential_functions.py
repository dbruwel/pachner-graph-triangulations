from typing import Optional, Type

import numpy as np
import regina
from snappy import Manifold


def is_knotted(t: regina.engine.Triangulation3) -> bool:
    if t.homology(1).rank() > 1:  # Super fast, but bad
        return True
    elif len(t.fundamentalGroup().relations()) > 0:  # Slowed but better
        return True
    elif not t.isSolidTorus():  # Super slow but perfect
        return True
    else:
        return False


class Score:
    pinch_first = True
    base_score = 0
    agg_func = None

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        raise NotImplementedError


class NormAlexanderPolynomial(Score):
    pinch_first = True
    base_score = 1

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        m = Manifold(tri)
        coeffs = m.alexander_polynomial().coefficients()
        score = np.dot(coeffs, coeffs)
        return score


class DegreeAlexanderPolynomial(Score):
    pinch_first = True
    base_score = 0

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        m = Manifold(tri)
        alex_poly = m.alexander_polynomial()
        score = alex_poly.degree() - alex_poly.valuation()
        return score


class DeterminantAlexanderPolynomial(Score):
    pinch_first = True
    base_score = 1

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        m = Manifold(tri)
        alex_poly = m.alexander_polynomial()
        score = np.abs(alex_poly(-1))
        return score


class AverageEdgeDegree(Score):
    pinch_first = False
    base_score = None

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        score = edge.degree()
        return score


class VarianceEdgeDegree(Score):
    pinch_first = False
    base_score = None
    agg_func = np.var

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        score = edge.degree()
        return score


class NumGenerators(Score):
    pinch_first = True
    base_score = 1

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        fg = tri.fundamentalGroup()
        score = fg.countGenerators()
        return score


class Potential:
    def __init__(self, potential: Type[Score], max_size: Optional[int] = 30):
        self.max_size = max_size
        self.potential = potential

    def calc_potential(
        self, iso: str
    ) -> tuple[float | np.floating, float | np.floating, int, bool]:
        scores = []
        knotted = []

        edges = regina.engine.Triangulation3.fromIsoSig(iso).countEdges()
        all_knoted = True

        if (not self.max_size is None) and (edges > self.max_size):
            return -np.inf, 0.0, 0, False

        for i in range(edges):
            tri = regina.engine.Triangulation3.fromIsoSig(iso)
            edge = tri.edge(i)
            if not self.potential.pinch_first:
                score = self.potential.potential(tri, edge)

            tri.pinchEdge(edge)

            if is_knotted(tri):
                if self.potential.pinch_first:
                    score = self.potential.potential(tri, edge)
                knotted.append(1)
            else:
                if self.potential.pinch_first:
                    score = self.potential.base_score
                knotted.append(0)
                all_knoted = False

            scores.append(score)  # type: ignore

        p_knotted = np.mean(knotted)
        count_unknotted = len(knotted) - np.sum(knotted)

        if self.potential.agg_func is not None:
            agg_score = self.potential.agg_func(scores)
        else:
            agg_score = np.mean(scores)

        return agg_score, p_knotted, count_unknotted, all_knoted


if __name__ == "__main__":
    iso = "cMcabbgqs"

    print(
        "NormAlexanderPolynomial",
        Potential(NormAlexanderPolynomial).calc_potential(iso)[0],
    )
    print(
        "DegreeAlexanderPolynomial",
        Potential(DegreeAlexanderPolynomial).calc_potential(iso)[0],
    )
    print(
        "DeterminantAlexanderPolynomial",
        Potential(DeterminantAlexanderPolynomial).calc_potential(iso)[0],
    )
    print("AverageEdgeDegree", Potential(AverageEdgeDegree).calc_potential(iso)[0])
    print("VarianceEdgeDegree", Potential(VarianceEdgeDegree).calc_potential(iso)[0])
    print("NumGenerators", Potential(NumGenerators).calc_potential(iso)[0])
