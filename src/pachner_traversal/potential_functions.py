import numpy as np
import regina  # type: ignore
from snappy import Manifold


def is_knotted(t):
    if t.homology(1).rank() > 1:  # Super fast, but bad
        return True
    elif len(t.fundamentalGroup().relations()) > 0:  # Slowed but better
        return True
    elif not t.isSolidTorus():  # Super slow but perfect
        return True
    else:
        return False


def norm_alexander_polynomial(tri):
    m = Manifold(tri)
    coeffs = m.alexander_polynomial().coefficients()
    score = np.dot(coeffs, coeffs)
    return score


def deg_alexander_polynomial(tri):
    m = Manifold(tri)
    alex_poly = m.alexander_polynomial()
    score = alex_poly.degree() - alex_poly.valuation()
    return score


def det_alexander_polynomial(tri):
    m = Manifold(tri)
    alex_poly = m.alexander_polynomial()
    score = np.abs(alex_poly.det(-1))
    return score


def norm_n_generators(tri):
    fg = tri.fundamentalGroup()
    score = fg.countGenerators()
    return score


def n_generators_pinched_potential(iso):
    scores = []
    knotted = []

    edges = regina.Triangulation3.fromIsoSig(iso).countEdges()
    all_knoted = True

    if edges > 30:
        return -np.inf, 0, 0, False

    for i in range(edges):
        tri = regina.Triangulation3.fromIsoSig(iso)
        edge = tri.edge(i)
        tri.pinchEdge(edge)

        if is_knotted(tri):
            score = norm_n_generators(tri)
            scores.append(score)
            knotted.append(1)
        else:
            scores.append(1)
            knotted.append(0)
            all_knoted = False

    p_knotted = np.mean(knotted)
    count_unknotted = len(knotted) - np.sum(knotted)
    average_score = np.mean(scores)

    return average_score, p_knotted, count_unknotted, all_knoted


def edge_degree_variance_potential(iso):
    edge_degrees = []
    knotted = []

    tri = regina.Triangulation3.fromIsoSig(iso)

    edges = tri.countEdges()
    all_knoted = True

    if edges > 30:
        return -np.inf, 0, 0, False

    for i in range(edges):
        tri = regina.Triangulation3.fromIsoSig(iso)
        edge = tri.edge(i)
        degree = edge.degree()
        edge_degrees.append(degree)

        tri.pinchEdge(edge)

        if is_knotted(tri):
            knotted.append(1)
        else:
            knotted.append(0)
            all_knoted = False

    p_knotted = np.mean(knotted)
    count_unknotted = len(knotted) - np.sum(knotted)
    edge_degree_variance = np.mean(edge_degrees)

    return edge_degree_variance, p_knotted, count_unknotted, all_knoted


def alexander_potential(iso):
    scores = []
    knotted = []

    edges = regina.Triangulation3.fromIsoSig(iso).countEdges()
    all_knoted = True

    if edges > 30:
        return -np.inf, 0, 0, False

    for i in range(edges):
        tri = regina.Triangulation3.fromIsoSig(iso)
        edge = tri.edge(i)
        tri.pinchEdge(edge)

        if is_knotted(tri):
            score = norm_alexander_polynomial(tri)
            scores.append(score)
            knotted.append(1)
        else:
            scores.append(1)
            knotted.append(0)
            all_knoted = False

    p_knotted = np.mean(knotted)
    count_unknotted = len(knotted) - np.sum(knotted)
    average_score = np.mean(scores)

    return average_score, p_knotted, count_unknotted, all_knoted
