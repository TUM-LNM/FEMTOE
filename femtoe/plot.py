import pyvista as pv
import numpy as np
import meshio
from .pure_bending import PureBending
from .shape_functions import evaluate_shape_function_derivatives
from . import assembly
from . import element
from .gauss_integration import gauss_integration
from IPython.display import Image, display


def get_displacement_mesh(problem: PureBending, nodal_displacements: np.ndarray):
    pyvista_mesh = pv.from_meshio(problem.mesh)
    pyvista_mesh.point_data["displacements"] = nodal_displacements.reshape((-1, 2))
    pyvista_mesh.point_data["displacements_3D"] = np.append(
        pyvista_mesh.point_data["displacements"],
        np.zeros((pyvista_mesh.point_data["displacements"].shape[0], 1)),
        axis=1,
    )

    return pyvista_mesh


def get_stresses_mesh(
    problem: PureBending,
    nodal_displacements: np.ndarray,
    use_pian_sumihara=False,
):
    stress_plot_cells = []
    stress_plot_nodes = []
    stress_plot_nodal_stresses = []
    stress_plot_nodal_displacements = []
    for cell in problem.cells:
        element_displacements = nodal_displacements[assembly.local_to_global(cell, 2)]

        stress_plot_cells.append([])
        for i, xi in enumerate(
            element.get_reference_nodal_coordinates(problem.cell_type)
        ):
            shape_function_derivatives = evaluate_shape_function_derivatives(
                problem.cell_type, xi
            )

            jacobian = shape_function_derivatives.T @ problem.nodes[cell]
            det_jacobian = np.linalg.det(jacobian)

            if det_jacobian <= 0:
                raise RuntimeError("Jacobian is not positive")

            dNdx = np.linalg.solve(jacobian, shape_function_derivatives.T)

            B_op = np.zeros((3, problem.num_dof_per_element))
            B_op[0, 0::2] = dNdx[0, :]
            B_op[1, 1::2] = dNdx[1, :]
            B_op[2, 0::2] = dNdx[1, :]
            B_op[2, 1::2] = dNdx[0, :]

            if use_pian_sumihara:
                P = np.array(
                    [
                        [1, 0, 0, xi[1], 0],
                        [0, 1, 0, 0, xi[0]],
                        [0, 0, 1, 0, 0],
                    ]
                )
                G, H = compute_g_h_matrices(problem, problem.nodes[cell])
                stress = P @ np.linalg.inv(H) @ G @ element_displacements
            else:
                stress = problem.constitutive_matrix @ B_op @ element_displacements

            stress_plot_cells[-1].append(len(stress_plot_nodes))
            stress_plot_nodes.append(
                problem.nodes[cell[i]],
            )
            stress_plot_nodal_stresses.append(stress)
            stress_plot_nodal_displacements.append(
                element_displacements[2 * i : 2 * i + 1]
            )

    stress_mesh = meshio.Mesh(
        np.array(stress_plot_nodes), {"quad": np.array(stress_plot_cells)}
    )
    stress_mesh.point_data["stress_xx"] = np.array(stress_plot_nodal_stresses)[:, 0]
    stress_mesh.point_data["stress_yy"] = np.array(stress_plot_nodal_stresses)[:, 1]
    stress_mesh.point_data["stress_xy"] = np.array(stress_plot_nodal_stresses)[:, 2]

    return pv.from_meshio(stress_mesh)


def plot(
    problem: PureBending,
    nodal_displacements: np.ndarray,
    scale_displacements: float = 1.0,
    use_pian_sumihara=False,
):
    my_plotter = pv.Plotter(shape=(4, 1), window_size=(1024, 2304), off_screen=True)
    my_plotter.clear_actors()

    # plot displacements
    my_plotter.subplot(0, 0)

    mesh_displacements = get_displacement_mesh(problem, nodal_displacements)
    warped_mesh = mesh_displacements.warp_by_vector(
        "displacements_3D", factor=scale_displacements
    )

    my_plotter.add_mesh(
        mesh_displacements, color="black", style="wireframe", line_width=2
    )
    my_plotter.add_mesh(
        warped_mesh,
        scalars="displacements",
        copy_mesh=True,
        show_edges=True,
        edge_color="black",
        line_width=2,
    )

    my_plotter.view_xy()
    my_plotter.show_axes()
    my_plotter.enable_2d_style()
    my_plotter.add_title(
        "Displacements (scaled by factor of " + str(scale_displacements) + ")"
    )

    # plot stresses
    mesh_stresses = get_stresses_mesh(problem, nodal_displacements, use_pian_sumihara)
    for i, stress_name in enumerate(["xx", "yy", "xy"]):
        my_plotter.subplot(i + 1, 0)
        my_plotter.add_mesh(
            mesh_stresses, color="black", style="wireframe", line_width=2
        )
        my_plotter.add_mesh(
            mesh_stresses,
            scalars="stress_" + stress_name,
            copy_mesh=True,
        )
        my_plotter.view_xy()
        my_plotter.show_axes()
        my_plotter.enable_2d_style()
        my_plotter.add_title("Stress " + stress_name)

    my_plotter.screenshot("plot.png")
    display(Image(filename="plot.png"))
    my_plotter.deep_clean()


def compute_g_h_matrices(problem: PureBending, nodal_coordinates: np.ndarray):
    num_stress_dofs = 5
    G = np.zeros((num_stress_dofs, problem.num_dof_per_element))
    H = np.zeros((num_stress_dofs, num_stress_dofs))

    # constitutive matrix of material
    Cinv = np.linalg.inv(problem.constitutive_matrix)

    # loop over Gauss points
    for gauss_weight, gauss_point in gauss_integration(problem.cell_type, 4):
        # setup ansatz for stresses
        P = np.array(
            [
                [1, 0, 0, gauss_point[1], 0],
                [0, 1, 0, 0, gauss_point[0]],
                [0, 0, 1, 0, 0],
            ]
        )

        # evaluates the shape functions at the Gauss point
        shape_function_derivatives = evaluate_shape_function_derivatives(
            problem.cell_type, gauss_point
        )

        # compute jacobian matrix and its determinant
        jacobian = shape_function_derivatives.T @ nodal_coordinates
        det_jacobian = np.linalg.det(jacobian)

        if det_jacobian <= 0:
            raise RuntimeError("Jacobian is not positive")

        # compute B-operator
        dNdx = np.linalg.solve(jacobian, shape_function_derivatives.T)

        B_op = np.zeros((3, problem.num_dof_per_element))
        B_op[0, 0::2] = dNdx[0, :]
        B_op[1, 1::2] = dNdx[1, :]
        B_op[2, 0::2] = dNdx[1, :]
        B_op[2, 1::2] = dNdx[0, :]

        # integrate Pian-Sumihara matrices
        G += P.T @ B_op * det_jacobian * gauss_weight
        H += P.T @ Cinv @ P * det_jacobian * gauss_weight

    return G, H
