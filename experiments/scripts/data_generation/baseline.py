"""Computes the baseline of acceptable strings given a random string.

WARNING: This is old code that forms a biased estimate.

This code generates a collection of acceptable triangulations and dehydrates them, it then uses these
to form a list of the avaliable characters for each position and then randomly samples from them and checks
if the resulting string is acceptable. The problem with this method is that the avaliable charecters are
only correct for "encodable" strings, that is strings that results from the actual dehydration of a triangulation,
not all parsable strings. Additionally, the method to determine the avaliable charecters at each step often
misses characters. A more accurate tree search method exists and is the primary refference point for these
statistics.
"""

import argparse
import random

from regina import Perm4, Triangulation3


def gen_acceptable(size: int) -> Triangulation3:
    """Generates an acceptable triangulation through random gluings.

    Parameters
    ----------
    size : int
        The size of the triangulation (number of tetrahedra)

    Returns
    -------
    Triangulation3
        The triangulation.
    """
    tri = Triangulation3()
    for _ in range(size):
        tri.newTetrahedron()

    open_faces = []
    for i in range(size):
        tet = tri.tetrahedron(i)
        for f in range(4):
            open_faces.append((tet, f))

    random.shuffle(open_faces)

    while open_faces:
        tet1, face1 = open_faces.pop()
        tet2, face2 = open_faces.pop()

        perm_list = [0, 0, 0, 0]
        perm_list[face1] = face2

        rem_verts1 = [v for v in range(4) if v != face1]
        rem_verts2 = [v for v in range(4) if v != face2]

        random.shuffle(rem_verts2)

        for i in range(3):
            perm_list[rem_verts1[i]] = rem_verts2[i]

        gluing = Perm4(perm_list[0], perm_list[1], perm_list[2], perm_list[3])
        tet1.join(face1, tet2, gluing)

    return tri


def is_manifold(tri: Triangulation3) -> bool:
    """Checks if a given triangulation is a manifold through a vertex link check.

    Parameters
    ----------
    tri : Triangulation3
        The triangulation to check.

    Returns
    -------
    bool
        If the triangulation is a valid manifold.
    """

    is_manifold = True
    for i in range(tri.countVertices()):
        v = tri.vertex(i)
        if v.linkEulerChar() != 2:
            is_manifold = False
            break

    return is_manifold


def gen_av_chars(size: int, N: int = 100_000) -> tuple[dict, int]:
    """Generates a dictionary of the avaliable charecters at each position for a valid dehydration signature.

    Parameters
    ----------
    size : int
        The size of the triangulation to consider.
    N : int, optional
        The number of samples to generate for the check., by default 100_000

    Returns
    -------
    tuple[dict, int]
        The dictionary of position charecter pairs, and the length of the signature.
    """
    sigs = []
    sig = ""

    for _ in range(N):
        tri = gen_acceptable(size)
        if tri.isConnected():
            sig = tri.dehydrate()
            sigs.append(sig)

    av_chars = {}
    seq_len = len(sig)

    for cid in range(seq_len):
        av_chars[cid] = list({sig[cid] for sig in sigs})

    return av_chars, seq_len


def get_acceptable(size: int, N: int = 100_000_000) -> float:
    """Estimates the proportion of random strings are valid dehydration signatures
    when each string is generated from the avaliable charecters at each position.

    Parameters
    ----------
    size : int
        The size of the triangulation to conisder.
    N : int, optional
        The number of strings to generate and check, by default 100_000_000

    Returns
    -------
    float
        The estimated proprtion.
    """

    acceptable_count = 0
    valid_count = 0

    av_chars, seq_len = gen_av_chars(size)

    for _ in range(N):
        word = "".join([random.choice(av_chars[cid]) for cid in range(seq_len)])

        try:
            t = Triangulation3.rehydrate(word)
            valid_count += 1 if t.isValid() else 0
            acceptable_count += 1
        except Exception:  # noqa: BLE001, S110
            pass

    p = acceptable_count / N
    return p


def main():
    parser = argparse.ArgumentParser(
        description="A script to estimate the random string generation efficiency."
    )
    parser.add_argument(
        "-s", "--size", type=int, default=10, help="Number of tetrahedra"
    )
    args = parser.parse_args()
    size = args.size
    print(f"startign {size}")

    p = get_acceptable(size)
    print(f"{size}: {p}")


if __name__ == "__main__":
    main()
