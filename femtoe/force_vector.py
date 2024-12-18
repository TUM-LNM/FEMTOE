from . import pure_bending
import numpy as np
from .gauss_integration import gauss_integration
from .shape_functions import (
    evaluate_shape_functions,
    evaluate_shape_function_derivatives,
)
from .element import get_transformation, number_of_element_nodes


def compute_element_force_vector_volume_forces(
    problem: pure_bending.PureBending, nodal_coordinates: np.ndarray
) -> np.ndarray:
    force_vector = np.zeros(problem.num_dof_per_element)

    # loop over Gauss points
    numgp = 4
    for gauss_weight, gauss_point in gauss_integration(problem.cell_type, numgp):
        shape_functions = evaluate_shape_functions(problem.cell_type, gauss_point)
        shape_function_derivatives = evaluate_shape_function_derivatives(
            problem.cell_type, gauss_point
        )

        x = shape_functions.dot(nodal_coordinates)

        N = np.zeros((problem.num_dof_per_node, problem.num_dof_per_element))
        N[0, 0::2] = shape_functions
        N[1, 1::2] = shape_functions

        body_force_at_gauss_point = problem.get_body_forces(x)

        jacobian = shape_function_derivatives.T @ nodal_coordinates
        det_jacobian = np.linalg.det(jacobian)

        force_vector -= N.T @ body_force_at_gauss_point * det_jacobian * gauss_weight

    return force_vector


def compute_element_force_vector_surface_traction(
    problem: pure_bending.PureBending, nodal_coordinates: np.ndarray
) -> np.ndarray:
    force_vector = np.zeros(problem.traction.num_dof_per_element)

    numgp = 2
    for gauss_weight, gauss_point in gauss_integration(
        problem.traction.cell_type, numgp
    ):
        shape_functions = evaluate_shape_functions(
            problem.traction.cell_type, gauss_point
        )
        shape_function_derivatives = evaluate_shape_function_derivatives(
            problem.traction.cell_type, gauss_point
        )

        N = np.zeros(
            (
                problem.num_dof_per_node,
                problem.num_dof_per_node
                * number_of_element_nodes(problem.traction.cell_type),
            )
        )

        for i in range(problem.num_dof_per_node):
            N[i, i :: problem.num_dof_per_node] = shape_functions

        jacobian = np.array(
            shape_function_derivatives.T
            @ nodal_coordinates
            @ get_transformation(problem.traction.cell_type, nodal_coordinates)
        )
        if len(jacobian.shape) == 0:
            det_jacobian = jacobian
        else:
            det_jacobian = np.linalg.det(jacobian)

        x = shape_functions.dot(nodal_coordinates)

        force_vector -= (
            N.T @ problem.traction.get_traction_vector(x) * det_jacobian * gauss_weight
        )

    return force_vector
