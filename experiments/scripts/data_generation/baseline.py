import argparse
import random

from regina import Perm4, Triangulation3


def gen_acceptable(size):
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


def isManifold(tri):
    is_manifold = True
    for i in range(tri.countVertices()):
        v = tri.vertex(i)
        if v.linkEulerChar() != 2:
            is_manifold = False
            break

    return is_manifold


def gen_av_chars(size, N=100_000):
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
        av_chars[cid] = list(set([sig[cid] for sig in sigs]))

    return av_chars, seq_len


def get_acceptable(size, N=100_000_000):
    acceptable_count = 0
    valid_count = 0

    av_chars, seq_len = gen_av_chars(size)

    for _ in range(N):
        word = "".join([random.choice(av_chars[cid]) for cid in range(seq_len)])

        try:
            t = Triangulation3.rehydrate(word)
            valid_count += 1 if t.isValid() else 0
            acceptable_count += 1
        except Exception:
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
