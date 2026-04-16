from __future__ import annotations

import logging
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from regina import Triangulation3, Triangulation4

logger = logging.getLogger(__name__)


# Error classes
class FindCellFailure(Exception):
    """Raised when a cell cannot be found in the poset."""


class FindLayerFailure(Exception):
    """Raised when a layer cannot be found in the poset."""


class LogicalMistake(Exception):
    """Raised when a logical error occurs in poset operations."""


class InputError(Exception):
    """Raised when invalid input is provided to a function."""


class PosetNode:
    """Represents a node in the face poset.

    PosetNodes have three main attributes: cell, name, and dim. They are uniquely
    identified inside the FacePoset by the tuple (self.dim, self.name). They also
    carry 'pointers' to other cells: parents and children.

    Note that during creation of the FacePoset, a parent may have multiple faces
    which are identified. As such, the PosetNode.parents list may well have repeat
    entries. This is intended behaviour. To strip these edges, apply
    FacePoset.strip_multi_edges() to the FacePoset object. Parent / child
    relationships will be stored in a separate 'irregular_parents' and
    'irregular_children' lists as attributes of each PosetNode.

    Attributes:
        cell: A reference to the Regina object for the cell in the triangulation.
        name: An integer index uniquely identifying this cell amongst cells of the same dimension.
        dim: An integer corresponding to the dimension of the cell.
        irregular_parents: List of irregular parent nodes.
        irregular_children: List of irregular child nodes.
        morse_matched: The matched node in Morse matching, or None.
        parents: List of parent nodes.
        children: List of child nodes.
    """

    def __init__(
        self,
        dim: int,
        name: int,
        cell: Any,
        parents: Optional[List[PosetNode]] = None,
        children: Optional[List[PosetNode]] = None,
    ) -> None:
        """Initializes a PosetNode.

        Args:
            dim: The dimension of the cell.
            name: The name of the cell.
            cell: The Regina cell object.
            parents: List of parent nodes. Defaults to None.
            children: List of child nodes. Defaults to None.
        """
        self.cell = cell
        self.name = name
        self.dim = dim
        self.irregular_parents: List[PosetNode] = []
        self.irregular_children: List[PosetNode] = []
        self.morse_matched: Optional[PosetNode] = None

        if not parents:
            self.parents: List[PosetNode] = []
        else:
            self.parents = parents
        if not children:
            self.children: List[PosetNode] = []
        else:
            self.children = children

    def key(self) -> Tuple[int, int]:
        """Returns the key tuple for the node.

        Returns:
            A tuple (dim, name).
        """
        return (self.dim, self.name)

    def __key(self) -> Tuple[int, int]:
        return (self.dim, self.name)

    def __hash__(self) -> int:
        return hash(self.__key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PosetNode):
            return NotImplemented
        return self.__key() == other.__key()

    def add_child(self, node: PosetNode) -> None:
        """Adds a child node.

        Args:
            node: The child node to add.
        """
        self.children.append(node)

    def add_parent(self, node: PosetNode) -> None:
        """Adds a parent node.

        Args:
            node: The parent node to add.
        """
        self.parents.append(node)

    def remove_child(self, node: PosetNode, error: bool = False) -> None:
        """Removes a child node.

        Args:
            node: The child node to remove.
            error: Whether to raise an error if the node is not found. Defaults to False.

        Raises:
            FindCellFailure: If error is True and the node is not found.
        """
        if node in self.irregular_children:
            self.irregular_children.remove(node)
        if node in self.children:
            self.children.remove(node)
        elif error:
            raise FindCellFailure(f"No such cell {node.dim, node.name} found")

    def remove_parent(self, node: PosetNode, error: bool = False) -> None:
        """Removes a parent node.

        Args:
            node: The parent node to remove.
            error: Whether to raise an error if the node is not found. Defaults to False.

        Raises:
            FindCellFailure: If error is True and the node is not found.
        """
        if node in self.irregular_parents:
            self.irregular_parents.remove(node)
        if node in self.parents:
            self.parents.remove(node)
        elif error:
            raise FindCellFailure(f"No such cell {node.dim, node.name} found")


class FacePoset:
    """Represents the face poset diagram.

    Nodes in the diagram are given by the PosetNode class. FacePoset can be
    instantiated by passing a TriangulationN (e.g., Triangulation3) in the Regina
    format, along with the dimension of the triangulation.

    This is designed so that the user should only interact with the FacePoset
    object, with the PosetNode objects used only for the backend of the FacePoset
    object.

    Example usage:

        tri = Triangulation3.fromIsoSig('fLAMcbcbdeehxjqhr')
        fp = FacePoset(triangulation=tri, dim=3)
        fp.strip_multi_edges()
        fp.output_poset()

    Attributes:
        layers: A dictionary mapping dimensions to dictionaries of nodes.
        dim: The dimension of the triangulation.
    """

    def __init__(
        self,
        triangulation: Optional[Union[Triangulation3, Triangulation4]] = None,
        dim: Optional[int] = None,
    ) -> None:
        """Initializes a FacePoset.

        Args:
            triangulation: The Regina triangulation object. Defaults to None.
            dim: The dimension of the triangulation. Defaults to None.

        Raises:
            InputError: If only one of triangulation or dim is provided.
        """
        self.layers: Dict[int, Dict[int, PosetNode]] = {}

        if (dim is None) ^ (triangulation is None):
            msg = "Must provide both a triangulation and its dimension, or neither of these inputs"
            raise InputError(msg)

        if (dim is not None) and (triangulation is not None):
            self.dim: int = dim
            for dimension in range(dim):
                self.layers[dimension] = {
                    name: PosetNode(dimension, name, cell)
                    for name, cell in enumerate(triangulation.faces(dimension))
                }
            self.layers[dim] = {
                name: PosetNode(dim, name, cell)
                for name, cell in enumerate(triangulation.simplices())
            }

            for dimension in range(dim, 0, -1):
                for name, node in self.layers[dimension].items():
                    cell = node.cell
                    for j in range(dimension + 1):
                        face = cell.face(dimension - 1, j)
                        for face_name, face_node in self.layers[dimension - 1].items():
                            if face_node.cell == face:
                                self.add_arc(
                                    (dimension, name), (dimension - 1, face_name)
                                )

    def get_node(self, dim: int, name: int) -> PosetNode:
        """Retrieves a node by dimension and name.

        Args:
            dim: The dimension of the node.
            name: The name of the node.

        Returns:
            The PosetNode object.

        Raises:
            FindCellFailure: If the node is not found.
        """
        try:
            return self.layers[dim][name]
        except KeyError:
            raise FindCellFailure(f"Could not find cell {name} in layer {dim}")

    def get_cell(self, dim: int, name: int) -> Any:
        """Retrieves the cell object by dimension and name.

        Args:
            dim: The dimension of the cell.
            name: The name of the cell.

        Returns:
            The cell object.
        """
        return self.layers[dim][name].cell

    def add_node(self, dim: int, name: int, cell: Any) -> None:
        """Adds a new node to the poset.

        Args:
            dim: The dimension of the node.
            name: The name of the node.
            cell: The cell object.
        """
        if dim not in self.layers:
            self.layers[dim] = {}
        self.layers[dim][name] = PosetNode(dim=dim, name=name, cell=cell)

    def remove_node(
        self, dim: int, node_label: int, suppress_error: bool = False
    ) -> None:
        """Removes a node from the poset.

        Args:
            dim: The dimension of the node.
            node_label: The name of the node.
            suppress_error: Whether to suppress errors if the node or layer is not found. Defaults to False.

        Raises:
            FindLayerFailure: If the layer is not found and suppress_error is False.
            FindCellFailure: If the node is not found and suppress_error is False.
        """
        if dim not in self.layers:
            if suppress_error:
                return
            raise FindLayerFailure(
                f"Could not find layer of dimension {dim} when deleting node {node_label}"
            )
        if node_label not in self.layers[dim]:
            if suppress_error:
                return
            raise FindCellFailure(
                f"Could not find node {(dim, node_label)} when trying to delete it"
            )
        node = self.layers[dim][node_label]
        for child in node.children:
            child.remove_parent(node)
        for irr_child in node.irregular_children:
            irr_child.remove_parent(node)
        for parent in node.parents:
            parent.remove_child(node)
        for irr_parent in node.irregular_parents:
            irr_parent.remove_child(node)
        del self.layers[dim][node_label]
        del node

    def add_arc(self, n1_tup: Tuple[int, int], n2_tup: Tuple[int, int]) -> None:
        """Adds an arc between two nodes.

        Args:
            n1_tup: Tuple (dim, name) of the first node.
            n2_tup: Tuple (dim, name) of the second node.

        Raises:
            FindLayerFailure: If a layer is not found.
            FindCellFailure: If a node is not found.
            LogicalMistake: If the nodes are in the same dimension or not adjacent.
        """
        dim1, name1 = n1_tup
        dim2, name2 = n2_tup

        for dim in (dim1, dim2):
            if dim not in self.layers:
                raise FindLayerFailure(
                    f"Could not find layer {dim} for creating arc between {n1_tup} and {n2_tup}"
                )

        for dim, name in (n1_tup, n2_tup):
            if name not in self.layers[dim]:
                raise FindCellFailure(
                    f"Could not find node ({dim}, {name}) for creating arc between {n1_tup} and {n2_tup}"
                )

        if dim1 == dim2:
            raise LogicalMistake(
                "Cannot place arc between two nodes of the same dimension"
            )

        if abs(dim1 - dim2) != 1:
            raise LogicalMistake(
                "Cannot place arc between two nodes which are not in adjacent layers"
            )

        if dim1 < dim2:
            dim1, name1, dim2, name2 = dim2, name2, dim1, name1

        # so after this point, the (dim1, name1) is the node with the higher dimension
        n1: PosetNode = self.layers[dim1][name1]
        n2: PosetNode = self.layers[dim2][name2]

        n1.add_child(n2)
        n2.add_parent(n1)

    def remove_arc(
        self, tup1: Tuple[int, int], tup2: Tuple[int, int], error: bool = False
    ) -> None:
        """Removes an arc between two nodes.

        Args:
            tup1: Tuple (dim, name) of the first node.
            tup2: Tuple (dim, name) of the second node.
            error: Whether to raise errors if the arc does not exist. Defaults to False.

        Raises:
            FindLayerFailure: If a layer is not found.
            FindCellFailure: If a node is not found.
            LogicalMistake: If the nodes are in the same dimension or not adjacent.
        """
        dim1, name1 = tup1
        dim2, name2 = tup2

        for dim in (dim1, dim2):
            if dim not in self.layers:
                raise FindLayerFailure(
                    f"Could not find layer {dim} for removing arc between {tup1} and {tup2}"
                )

        for dim, name in (tup1, tup2):
            if name not in self.layers[dim]:
                raise FindCellFailure(
                    f"Could not find node ({dim}, {name}) for removing arc between {tup1} and {tup2}"
                )

        if dim1 == dim2:
            raise LogicalMistake(
                "Arc cannot exist between two nodes of the same dimension"
            )

        if abs(dim1 - dim2) != 1:
            raise LogicalMistake(
                "Arc cannot go between two nodes which are not in adjacent layers"
            )

        if dim1 < dim2:
            dim1, name1, dim2, name2 = dim2, name2, dim1, name1

        # so after this point, the (dim1, name1) is the node with the higher dimension
        n1: PosetNode = self.layers[dim1][name1]
        n2: PosetNode = self.layers[dim2][name2]

        if error:
            n1.remove_child(n2, error=True)
            n2.remove_parent(n1, error=True)
        else:
            n1.remove_child(n2)
            n2.remove_parent(n1)

    def output_poset(self) -> None:
        """Outputs the face poset diagram to the logger.

        This depicts the face poset diagram from the 0th dimension cells upwards.
        Arcs are expressed downwards.
        """
        title = "This depicts the face poset diagram from the 0th dimension cells upwards. Arcs are expressed downwards."
        logger.info(title)
        for dim in sorted(self.layers.keys()):
            logger.info(f"Dim {dim}:")
            if dim == 0:
                for cell_name in self.layers[dim]:
                    logger.info(cell_name)
            else:
                for cell_name in self.layers[dim]:
                    children = self.layers[dim][cell_name].children
                    if not children:
                        logger.info(cell_name)
                    else:
                        string = ", ".join([str(c.name) for c in children])
                        logger.info(f"{cell_name}: {string}")

    def strip_multi_edges(self) -> None:
        """Strips multi-edges from the poset.

        Moves duplicate edges to irregular lists.
        """
        for dimension in range(self.dim, 0, -1):
            for node in self.layers[dimension].values():
                res = self.separate_duplicates(node.children)
                node.children, node.irregular_children = res
        for dimension in range(self.dim):
            for node in self.layers[dimension].values():
                res = self.separate_duplicates(node.parents)
                node.parents, node.irregular_parents = res

    def separate_duplicates(
        self, array: List[PosetNode]
    ) -> Tuple[List[PosetNode], List[PosetNode]]:
        """Separates duplicates from a list of nodes.

        Args:
            array: List of PosetNode objects.

        Returns:
            A tuple of (not_duplicates, duplicates).
        """
        seen: Dict[PosetNode, int] = {}
        for x in array:
            seen[x] = seen.get(x, 0) + 1

        not_duplicates = [el for el, count in seen.items() if count == 1]
        duplicates = [el for el, count in seen.items() if count > 1]
        return not_duplicates, duplicates

    def match(self, tup1: Tuple[int, int], tup2: Tuple[int, int]) -> None:
        """Matches two nodes for Morse matching.

        Args:
            tup1: Tuple (dim, name) of the first node.
            tup2: Tuple (dim, name) of the second node.

        Raises:
            FindLayerFailure: If a layer is not found.
            FindCellFailure: If a node is not found.
            LogicalMistake: If the nodes are in the same dimension or not adjacent.
        """
        dim1, name1 = tup1
        dim2, name2 = tup2

        for dim in (dim1, dim2):
            if dim not in self.layers:
                raise FindLayerFailure(
                    f"Could not find layer {dim} for matching {tup1} and {tup2}"
                )

        for dim, name in (tup1, tup2):
            if name not in self.layers[dim]:
                raise FindCellFailure(
                    f"Could not find node ({dim}, {name}) for matching {tup1} and {tup2}"
                )

        if dim1 == dim2:
            raise LogicalMistake("Cannot match two nodes of the same dimension")

        if abs(dim1 - dim2) != 1:
            raise LogicalMistake(
                "Cannot match two nodes which are not in adjacent layers"
            )

        n1: PosetNode = self.layers[dim1][name1]
        n2: PosetNode = self.layers[dim2][name2]

        n1.morse_matched = n2
        n2.morse_matched = n1

    def randomised_morse_matching(
        self, randomise: bool = True
    ) -> Tuple[List[Tuple[Tuple[int, int], Tuple[int, int]]], List[Tuple[int, int]]]:
        """Performs randomised Morse matching on the poset.

        This is implemented in a destructive way, modifying the poset.

        Args:
            randomise: Whether to randomise the order of nodes. Defaults to True.

        Returns:
            A tuple of (morse_pairs, critical), where morse_pairs is a list of matched pairs
            and critical is a list of critical cells.
        """
        morse_pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        critical: List[Tuple[int, int]] = []
        while True:
            critical_candidate: Optional[PosetNode] = None
            match_made = False
            for dimension in range(self.dim, -1, -1):
                nodes = list(self.layers[dimension].items())
                if randomise:
                    random.shuffle(nodes)
                for name, node in nodes:
                    if name not in self.layers[dimension]:
                        continue

                    if critical_candidate is None:
                        critical_candidate = node

                    if len(node.parents) == 1 and not node.irregular_parents:
                        parent = node.parents[0]
                        morse_pairs.append((node.key(), parent.key()))

                        self.remove_node(parent.dim, parent.name)
                        self.remove_node(node.dim, node.name)
                        match_made = True
            # need to save a candidate for making a cell critical so I don't need to repeat the search
            if not match_made and critical_candidate is not None:
                # make cell critical
                critical.append((critical_candidate.dim, critical_candidate.name))
                self.remove_node(critical_candidate.dim, critical_candidate.name)
            if critical_candidate is None:
                break
        return morse_pairs, critical

    def keys(self, array: List[PosetNode]) -> List[Tuple[int, int]]:
        """Returns the keys of a list of nodes.

        Args:
            array: List of PosetNode objects.

        Returns:
            List of (dim, name) tuples.
        """
        return [el.key() for el in array]


def estimate_critical_count(iso: str, dim: int = 3, itts: int = 100) -> float:
    """Estimates the minimum number of critical cells in each dimension for a triangulation.

    Args:
        iso: The isomorphism signature of the triangulation.
        dim: The dimension of the triangulation. Defaults to 3.
        itts: The number of iterations to perform. Defaults to 100.

    Returns:
        The estimated minimum number of critical cells in the triangulation.
    """
    if dim == 3:
        triangulation = Triangulation3.fromIsoSig(iso)
    elif dim == 4:
        triangulation = Triangulation4.fromIsoSig(iso)
    else:
        raise InputError(
            "Only dimensions 3 and 4 are supported for estimating critical count"
        )

    fp = FacePoset(triangulation=triangulation, dim=dim)
    fp.strip_multi_edges()

    min_critical = float("inf")

    for _ in range(itts):
        _, critical = fp.randomised_morse_matching()
        count_critical = len(critical)

        if count_critical < min_critical:
            msg = f"New minimum critical count of {count_critical} found"
            logger.debug(msg)
            min_critical = count_critical

    return min_critical


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.setrecursionlimit(100)

    random.seed(42)
    iso = "DLLLPPvMvMMPwAvMQQLMcacfgeggklpmqqpsorrxvvzwyxyABCCbggwvooxxarpagagjbhaaggjcbvjxf"
    min_critical = float("inf")
    counts_critical3 = []
    counts_critical2 = []
    counts_critical1 = []
    counts_critical0 = []

    for i in range(1000):
        tri = Triangulation3.fromIsoSig(iso)
        fp = FacePoset(triangulation=tri, dim=3)
        fp.strip_multi_edges()

        morse, critical = fp.randomised_morse_matching()
        count_critical = len(critical)
        counts_critical3.append(len([c for c in critical if c[0] == 3]))
        counts_critical2.append(len([c for c in critical if c[0] == 2]))
        counts_critical1.append(len([c for c in critical if c[0] == 1]))
        counts_critical0.append(len([c for c in critical if c[0] == 0]))

        if count_critical < min_critical:
            msg = f"New minimum critical count of {count_critical} found"
            logger.info(msg)
            min_critical = count_critical

    sig3 = np.mean(counts_critical3)
    sig2 = np.mean(counts_critical2)
    sig1 = np.mean(counts_critical1)
    sig0 = np.mean(counts_critical0)

    sig = np.array([sig0, sig1, sig2, sig3])
    logger.info(sig.round(2))
