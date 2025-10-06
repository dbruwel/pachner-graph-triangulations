from typing import Optional, Type

import numpy as np
import regina
from snappy import Manifold


def is_knotted(t: regina.engine.Triangulation3) -> bool:
    return not t.isSolidTorus()


class Score:
    pinch_first = True
    base_score = 0
    agg_func = None
    print_name = "score"

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        raise NotImplementedError


class NormAlexanderPolynomial(Score):
    pinch_first = True
    base_score = 1
    print_name = "norm_alex_poly"

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
    print_name = "deg_alex_poly"

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
    print_name = "det_alex_poly"

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
    print_name = "avg_edge_deg"

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
    print_name = "var_edge_deg"

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        score = edge.degree()
        return score


class NumGenerators(Score):
    pinch_first = True
    base_score = 1
    print_name = "num_gens"

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        fg = tri.fundamentalGroup()
        score = fg.countGenerators()
        return score

class KnottedFrac(Score):
    pinch_first = True
    base_score = 0
    print_name = "knotted_frac"

    @staticmethod
    def potential(
        tri: regina.engine.Triangulation3, edge: regina.engine.Face3_1
    ) -> float:
        return 1


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


def calc_composite_potential(iso):
    scores_alex_norm = []
    scores_alex_deg = []
    scores_alex_det = []
    scores_edge_var = []
    scores_num_gen = []

    knotted = []

    edges = regina.engine.Triangulation3.fromIsoSig(iso).countEdges()
    all_knoted = True

    for i in range(edges):
        tri = regina.engine.Triangulation3.fromIsoSig(iso)
        edge = tri.edge(i)
        score_edge_var = VarianceEdgeDegree.potential(tri, edge)

        tri.pinchEdge(edge)
        try:
            if is_knotted(tri):
                m = Manifold(tri)
                alex_poly = m.alexander_polynomial()
                coeffs = alex_poly.coefficients()

                score_alex_norm = np.dot(coeffs, coeffs)
                score_alex_deg = alex_poly.degree() - alex_poly.valuation()
                score_alex_det = np.abs(alex_poly(-1))
                score_num_gen = NumGenerators.potential(tri, edge)
                knotted.append(1)
            else:
                score_alex_norm = NormAlexanderPolynomial.base_score
                score_alex_deg = DegreeAlexanderPolynomial.base_score
                score_alex_det = DeterminantAlexanderPolynomial.base_score
                score_num_gen = NumGenerators.base_score
                knotted.append(0)
                all_knoted = False
        except Exception as e:
            import pdb

            pdb.set_trace()
            raise RuntimeError()

        scores_alex_norm.append(score_alex_norm)
        scores_alex_deg.append(score_alex_deg)
        scores_alex_det.append(score_alex_det)
        scores_edge_var.append(score_edge_var)
        scores_num_gen.append(score_num_gen)

    agg_score_alex_norm = np.mean(scores_alex_norm)
    agg_score_alex_deg = np.mean(scores_alex_deg)
    agg_score_alex_det = np.mean(scores_alex_det)
    agg_score_edge_var = np.var(scores_edge_var)
    agg_score_num_gen = np.mean(scores_num_gen)

    p_knotted = np.mean(knotted)
    count_unknotted = len(knotted) - np.sum(knotted)

    return (
        agg_score_alex_norm,
        agg_score_alex_deg,
        agg_score_alex_det,
        agg_score_edge_var,
        agg_score_num_gen,
        p_knotted,
        count_unknotted,
        all_knoted,
    )


if __name__ == "__main__":
    # iso = "cMcabbgqs"
    iso = "pLLLLPMvQAQbcgihjjhnklmmoooarewvdafxfcwfxax"

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
