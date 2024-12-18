import numpy as np


def number_of_element_nodes(cell_type: str):
    if cell_type == "quad":
        return 4

    if cell_type == "line":
        return 2

    raise NotImplementedError(f"Unsupported cell type {cell_type}")


def get_faces(cell_type: str, cell: np.ndarray) -> np.ndarray:
    if cell_type == "quad":
        return np.array(
            [
                [cell[0], cell[1]],
                [cell[1], cell[2]],
                [cell[2], cell[3]],
                [cell[3], cell[0]],
            ]
        )

    raise NotImplementedError(f"Unsupported cell type {cell_type}")


def get_transformation(cell_type: str, nodes: np.ndarray) -> np.ndarray:
    if cell_type == "line":
        transform = nodes[1] - nodes[0]
        transform /= np.linalg.norm(transform)
        return transform
    raise NotImplementedError(f"Unsupported cell type {cell_type}")


def get_reference_nodal_coordinates(cell_type: str):
    if cell_type == "quad":
        return np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
            ]
        )

    if cell_type == "line":
        return np.array(
            [
                [-1],
                [1],
            ]
        )

    raise NotImplementedError(f"Unsupported cell type {cell_type}")
