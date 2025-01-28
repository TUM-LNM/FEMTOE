import numpy as np
from .assembly import assemble_matrix
from .gauss_integration import gauss_integration
from .shape_functions import evaluate_shape_function_derivatives


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


def evaluate_global_stiffness_matrix(problem) -> np.ndarray:
    global_stiffness_matrix = np.zeros((problem.num_dof, problem.num_dof))

    for cell in problem.cells:
        # evaluate the element stiffness matrix
        stiffness_matrix = compute_element_stiffness_matrix(
            problem, problem.nodes[cell]
        )

        # assemble the element matrix into our global matrix
        assemble_matrix(
            global_stiffness_matrix, stiffness_matrix, cell, problem.num_dof_per_node
        )

    # apply dirichlet boundary conditions
    dirichlet_dofs = problem.dirichlet_dofs
    free_dofs = np.delete(np.arange(problem.num_dof), dirichlet_dofs)

    # delete respective colums and rows from our global stiffness matrix and force vector
    global_stiffness_matrix = global_stiffness_matrix[np.ix_(free_dofs, free_dofs)]

    return global_stiffness_matrix


def compute_element_stiffness_matrix(
    problem, nodal_coordinates: np.ndarray
) -> np.ndarray:
    # initialize empty stiffness matrix for integration
    element_stiffness_matrix = np.zeros(
        (problem.num_dof_per_element, problem.num_dof_per_element)
    )

    # constitutive matrix of material
    C = problem.constitutive_matrix

    # loop over Gauss points
    for gauss_weight, gauss_point in gauss_integration(problem.cell_type, 4):
        # evaluates the shape functions at the Gauss point
        shape_function_derivatives = evaluate_shape_function_derivatives(
            problem.cell_type, gauss_point
        )

        # compute jacobian matrix and its determinant (Note, this function will be implemented later)
        jacobian = shape_function_derivatives.T @ nodal_coordinates
        det_jacobian = np.linalg.det(jacobian)

        if det_jacobian <= 0:
            raise RuntimeError("Jacobian is not positive")

        dNdx = np.linalg.solve(jacobian, shape_function_derivatives.T)

        # compute B-operator
        B_op = np.zeros((3, problem.num_dof_per_element))
        B_op[0, 0::2] = dNdx[0, :]
        B_op[1, 1::2] = dNdx[1, :]
        B_op[2, 0::2] = dNdx[1, :]
        B_op[2, 1::2] = dNdx[0, :]

        # compute element stiffness matrix
        element_stiffness_matrix += B_op.T @ C @ B_op * det_jacobian * gauss_weight

    return element_stiffness_matrix
