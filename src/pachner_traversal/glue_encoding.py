import networkx as nx
import numpy as np
from regina import Perm4, Triangulation3
from scipy.sparse.csgraph import connected_components

# The following are generated with a simple loop
# ```python
# for i in range(12):
#     for j in range(12):
#         if i == j:
#             continue
#         i_tet_id, i_face_id, i_vertex_id = idx_to_pos(i)
#         j_tet_id, j_face_id, j_vertex_id = idx_to_pos(j)
#         if i_face_id == j_face_id:
#             face_block[i, j] = 1
#         if i_vertex_id == j_vertex_id:
#             vertex_block[i, j] = 1
# ```
face_block = np.array(
    [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    ]
)
vertex_block = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]
)

vertex_to_idx = {
    0: {1: 0, 2: 1, 3: 2},
    1: {0: 0, 2: 1, 3: 2},
    2: {0: 0, 1: 1, 3: 2},
    3: {0: 0, 1: 1, 2: 2},
}

idx_to_vertex = {
    0: {0: 1, 1: 2, 2: 3},
    1: {0: 0, 1: 2, 2: 3},
    2: {0: 0, 1: 1, 2: 3},
    3: {0: 0, 1: 1, 2: 2},
}


def pos_to_idx(tet_id, face_id, vertex_id):
    """Convert a position in the triangulation to an index for the gluing matrix."""
    return 12 * tet_id + 3 * face_id + vertex_to_idx[face_id][vertex_id]


def idx_to_pos(idx):
    """Convert an index in the gluing matrix back to a position in the triangulation."""
    tet_id = idx // 12
    face_id = (idx % 12) // 3
    vertex_idx = (idx % 12) % 3
    vertex_id = idx_to_vertex[face_id][vertex_idx]
    return (tet_id, face_id, vertex_id)


def tri_to_gluing(tri):
    """Convert a triangulation to a gluing matrix."""
    tets = [tet for tet in tri.simplices()]

    gluing_matrix = np.zeros((len(tets) * 12, len(tets) * 12), dtype=int)

    for source_tet_id, tet in enumerate(tets):
        for source_face_id in range(4):
            dest_tet_id = tet.adjacentSimplex(source_face_id).index()
            gluing = tet.adjacentGluing(source_face_id)
            dest_face_id = gluing[source_face_id]
            for source_vertex_id in vertex_to_idx[source_face_id]:
                dest_vertex_id = gluing[source_vertex_id]
                source_idx = pos_to_idx(source_tet_id, source_face_id, source_vertex_id)
                dest_idx = pos_to_idx(dest_tet_id, dest_face_id, dest_vertex_id)
                gluing_matrix[source_idx, dest_idx] += 1

    return gluing_matrix


def extract_perm(vertex_map) -> tuple[int, int, Perm4]:
    """Helper function to extract the tetrahedron ID, face ID, and permutation from a vertex map."""
    tet_id: int | None = None
    face_id: int | None = None
    for v in vertex_map:
        if tet_id is None:
            tet_id = vertex_map[v][0]
        elif tet_id != vertex_map[v][0]:
            raise ValueError("Inconsistent tetrahedron ID")
        if face_id is None:
            face_id = vertex_map[v][1]
        elif face_id != vertex_map[v][1]:
            raise ValueError("Inconsistent face ID")

    if tet_id is None or face_id is None:
        raise ValueError("Missing tetrahedron or face ID")

    perm: list[int | None] = [None] * 4
    for i in range(4):
        if i in vertex_map:
            v: int = vertex_map[i][2]
            perm[i] = v
        else:
            perm[i] = face_id

    if not set(perm) == set(range(4)):
        raise ValueError("Invalid permutation")

    return tet_id, face_id, Perm4(perm)


def gluing_to_tri(gluing_matrix):
    """Convert a gluing matrix back to a triangulation."""
    n_tets = gluing_matrix.shape[0] // 12

    glued = set()
    gluing_table = []
    for source_tet_id in range(n_tets):
        for source_face_id in range(4):
            vertex_map = {}
            for source_vertex_id in vertex_to_idx[source_face_id]:
                source_idx = pos_to_idx(source_tet_id, source_face_id, source_vertex_id)
                dest_idx = np.argmax(gluing_matrix[source_idx, :])
                vertex_map[source_vertex_id] = idx_to_pos(dest_idx)

            destination_tet_id, destination_face_id, perm = extract_perm(vertex_map)
            if (source_tet_id, source_face_id) in glued or (
                destination_tet_id,
                destination_face_id,
            ) in glued:
                continue
            gluing_table.append(
                (int(source_tet_id), int(source_face_id), int(destination_tet_id), perm)
            )
            glued.add((source_tet_id, source_face_id))
            glued.add((destination_tet_id, destination_face_id))

    t_recon = Triangulation3.fromGluings(n_tets, gluing_table)

    return t_recon


def get_face_graph(n_tet):
    data = [[np.zeros((12, 12), dtype=int) for _ in range(n_tet)] for _ in range(n_tet)]
    for i in range(n_tet):
        data[i][i] = face_block

    face_encoding = np.block(data)
    return face_encoding


def get_vertex_graph(n_tet):
    data = [[np.zeros((12, 12), dtype=int) for _ in range(n_tet)] for _ in range(n_tet)]
    for i in range(n_tet):
        data[i][i] = vertex_block

    vertex_encoding = np.block(data)
    return vertex_encoding


def encode(t):
    t_size = t.size()
    block_size = t_size * 12

    gluing_matrix = tri_to_gluing(t)
    face_graph = get_face_graph(t_size)
    vertex_graph = get_vertex_graph(t_size)

    full_adjacency = (
        np.block(
            [
                [
                    gluing_matrix + gluing_matrix,
                    gluing_matrix + face_graph,
                    gluing_matrix + vertex_graph,
                ],
                [
                    face_graph + gluing_matrix,
                    face_graph + face_graph,
                    face_graph + vertex_graph,
                ],
                [
                    vertex_graph + gluing_matrix,
                    vertex_graph + face_graph,
                    vertex_graph + vertex_graph,
                ],
            ]
        )
        / 2
    )

    D_m12 = np.diag(1 / np.sqrt(full_adjacency.sum(axis=0)))
    full_laplacian = np.eye(full_adjacency.shape[0]) - D_m12 @ full_adjacency @ D_m12

    evals, evecs = np.linalg.eigh(full_laplacian)
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    evecs_loaded = evecs @ np.diag(np.sqrt(evals))

    glue_encoding = evecs_loaded[: 1 * block_size]
    face_encoding = evecs_loaded[block_size : 2 * block_size]
    vertex_encoding = evecs_loaded[2 * block_size :]

    encoding = np.hstack([glue_encoding, face_encoding, vertex_encoding])
    return encoding


def joint_graph_matching(nd_face, target_face, nd_vertex, target_vertex):
    N = nd_face.shape[0]

    nd_f = (nd_face > 0.5).astype(int)
    nd_v = (nd_vertex > 0.5).astype(int)
    tgt_f = (target_face > 0.5).astype(int)
    tgt_v = (target_vertex > 0.5).astype(int)

    def get_components(labels):
        comps = {}
        for node, label in enumerate(labels):
            if label not in comps:
                comps[label] = []
            comps[label].append(node)
        return [comps[l] for l in sorted(comps.keys())]

    _, nd_tri_labels = connected_components(nd_f, directed=False)
    _, tgt_tri_labels = connected_components(tgt_f, directed=False)

    nd_tris = get_components(nd_tri_labels)
    tgt_tris = get_components(tgt_tri_labels)

    n_tris = len(nd_tris)

    nd_node_to_tri = {node: idx for idx, tri in enumerate(nd_tris) for node in tri}
    tgt_node_to_tri = {node: idx for idx, tri in enumerate(tgt_tris) for node in tri}

    nd_super = np.zeros((n_tris, n_tris), dtype=int)
    tgt_super = np.zeros((n_tris, n_tris), dtype=int)

    for u, v in np.argwhere(nd_v > 0):
        tri_u, tri_v = nd_node_to_tri[u], nd_node_to_tri[v]
        if tri_u != tri_v:
            nd_super[tri_u, tri_v] = 1

    for u, v in np.argwhere(tgt_v > 0):
        tri_u, tri_v = tgt_node_to_tri[u], tgt_node_to_tri[v]
        if tri_u != tri_v:
            tgt_super[tri_u, tri_v] = 1

    _, nd_tetra_labels = connected_components(nd_super, directed=False)
    _, tgt_tetra_labels = connected_components(tgt_super, directed=False)

    nd_tetras = get_components(nd_tetra_labels)
    tgt_tetras = get_components(tgt_tetra_labels)

    P = np.zeros((N, N), dtype=int)

    for tetra_idx in range(len(nd_tetras)):
        nd_t_indices = nd_tetras[tetra_idx]
        tgt_t_indices = tgt_tetras[tetra_idx]

        for i in range(4):
            tgt_tri_idx = tgt_t_indices[i]
            nd_tri_idx = nd_t_indices[i]

            tgt_nodes = tgt_tris[tgt_tri_idx]
            nd_nodes = nd_tris[nd_tri_idx]

            for tgt_node in tgt_nodes:
                tgt_neighbors = np.where(tgt_v[tgt_node] > 0)[0]
                tgt_neighbor_tris = {
                    tgt_node_to_tri[nb]
                    for nb in tgt_neighbors
                    if tgt_node_to_tri[nb] != tgt_tri_idx
                }

                tgt_signature = {tgt_t_indices.index(t) for t in tgt_neighbor_tris}

                for nd_node in nd_nodes:
                    nd_neighbors = np.where(nd_v[nd_node] > 0)[0]
                    nd_neighbor_tris = {
                        nd_node_to_tri[nb]
                        for nb in nd_neighbors
                        if nd_node_to_tri[nb] != nd_tri_idx
                    }

                    nd_signature = {nd_t_indices.index(t) for t in nd_neighbor_tris}

                    if tgt_signature == nd_signature:
                        P[tgt_node, nd_node] = 1
                        break

    return P


def project_gluing(n_gluing_matrix):
    n = n_gluing_matrix.shape[0]

    G = nx.Graph()

    for i in range(n):
        for j in range(i + 1, n):
            weight = n_gluing_matrix[i, j] + n_gluing_matrix[j, i]
            G.add_edge(i, j, weight=weight)

    matching = nx.max_weight_matching(G, maxcardinality=True)

    gluing_matrix = np.zeros((n, n))
    for u, v in matching:
        gluing_matrix[u, v] = 1
        gluing_matrix[v, u] = 1
    return gluing_matrix


def decode(encoding):
    n_vert = encoding.shape[0]
    n_tet = n_vert // 12
    n_latent = n_vert * 3

    glue_encoding = encoding[:, :n_latent]
    face_encoding = encoding[:, n_latent : 2 * n_latent]
    vertex_encoding = encoding[:, 2 * n_latent :]

    evecs_loaded = np.vstack([glue_encoding, face_encoding, vertex_encoding])

    full_laplacian = evecs_loaded @ evecs_loaded.T

    D = np.diag(
        np.concatenate([np.ones(n_vert) * 2, np.ones(n_vert * 2) * np.sqrt(5.5)])
    )

    full_adjacency = D @ (np.eye(full_laplacian.shape[0]) - full_laplacian) @ D

    nd_gluing_matrix = full_adjacency[:n_vert, :n_vert]
    nd_face_graph = full_adjacency[n_vert : 2 * n_vert, n_vert : 2 * n_vert]
    nd_vertex_graph = full_adjacency[2 * n_vert :, 2 * n_vert :]

    target_face_graph = get_face_graph(n_tet)
    target_vertex_graph = get_vertex_graph(n_tet)

    P_final = joint_graph_matching(
        nd_face_graph,
        target_face_graph,
        nd_vertex_graph,
        target_vertex_graph,
    )

    n_gluing_matrix = P_final @ nd_gluing_matrix @ P_final.T
    gluing_matrix = project_gluing(n_gluing_matrix)

    tri = gluing_to_tri(gluing_matrix)

    return tri


if __name__ == "__main__":
    t = Triangulation3.fromIsoSig("cMcabbgqs")
    gluing_matrix = tri_to_gluing(t)
    t_recon = gluing_to_tri(gluing_matrix)

    assert t.isIsomorphicTo(t_recon)
