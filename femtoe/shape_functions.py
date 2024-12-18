import numpy as np


def evaluate_shape_functions(cell_type: str, xi: np.ndarray):
    if cell_type == "quad":
        N = np.array(
            [
                0.25 * (1 - xi[0]) * (1 - xi[1]),
                0.25 * (1 + xi[0]) * (1 - xi[1]),
                0.25 * (1 + xi[0]) * (1 + xi[1]),
                0.25 * (1 - xi[0]) * (1 + xi[1]),
            ]
        )

        return N

    if cell_type == "line":
        N = np.array(
            [
                0.5 * (1 - xi),
                0.5 * (1 + xi),
            ]
        )

        return N

    raise NotImplementedError(f"Unsupported cell type {cell_type}")


def evaluate_shape_function_derivatives(cell_type: str, xi: np.ndarray):
    if cell_type == "quad":
        dN_dxi = np.array(
            [
                [-0.25 * (1 - xi[1]), -0.25 * (1 - xi[0])],
                [0.25 * (1 - xi[1]), -0.25 * (1 + xi[0])],
                [0.25 * (1 + xi[1]), 0.25 * (1 + xi[0])],
                [-0.25 * (1 + xi[1]), 0.25 * (1 - xi[0])],
            ]
        )

        return dN_dxi

    if cell_type == "line":
        dN_dxi = np.array([-0.5, 0.5])

        return dN_dxi

    raise NotImplementedError(f"Unsupported cell type {cell_type}")
