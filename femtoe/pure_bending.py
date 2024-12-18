import numpy as np
import meshio
from typing import List, Tuple, Callable

from .element import number_of_element_nodes, get_faces
from .problem import Problem


class PureBending:

    def __init__(
        self,
        *,
        width: float,
        height: float,
        num_ele_x,
        num_ele_y,
        youngs_modulus: float,
        nu: float
    ):
        self.mesh = get_mesh(num_ele_x=num_ele_x, num_ele_y=num_ele_y)

        self.youngs_modulus = youngs_modulus
        self.nu = nu

        self.traction = Traction(
            self.mesh, lambda x: 100e6 * np.array([1, 0]) * (-1 + 2 * x[1] / height)
        )

    @property
    def cell_type(self):
        return "quad"

    @property
    def cells(self):
        return self.mesh.cells_dict[self.cell_type]

    @property
    def nodes(self):
        return self.mesh.points

    @property
    def num_dof_per_node(self):
        return 2

    @property
    def num_dof_per_element(self):
        return self.num_dof_per_node * number_of_element_nodes(self.cell_type)

    @property
    def num_dof(self):
        return 2 * len(self.nodes)

    @property
    def constitutive_matrix(self):
        # plain strain
        return (
            self.youngs_modulus
            / ((1 + self.nu) * (1 - 2 * self.nu))
            * np.array(
                [
                    [1 - self.nu, self.nu, 0],
                    [self.nu, 1 - self.nu, 0],
                    [0, 0, (1 - 2 * self.nu) / 2],
                ]
            )
        )

    def get_body_forces(self, x: np.ndarray):
        return np.array([0, 0])

    @property
    def dirichlet_dofs(self):
        # the left side at x=0 is fixed in all directions
        node_ids = np.argwhere(np.abs(self.nodes @ np.array([1, 0])) < 1e-20)
        return np.vstack([2 * node_ids, 2 * node_ids + 1]).flatten()


class Traction:
    def __init__(
        self, mesh: meshio.Mesh, traction_function: Callable[[np.ndarray], np.ndarray]
    ):
        self.cells = mesh.cells_dict[self.cell_type]
        self.traction_function = traction_function

    @property
    def cell_type(self):
        return "line"

    @property
    def num_dof_per_node(self):
        return 2

    @property
    def num_dof_per_element(self):
        return self.num_dof_per_node * number_of_element_nodes(self.cell_type)

    def get_traction_vector(self, x: np.ndarray):
        return self.traction_function(x)


def get_mesh(length=0.4, height=0.1, num_ele_x=4, num_ele_y=1):
    x_coords = np.linspace(0.0, length, num_ele_x + 1)
    y_coords = np.linspace(0, height, num_ele_y + 1)

    points = []
    for x in x_coords:
        for y in y_coords:
            points.append(np.array([x, y]))

    points = np.array(points)

    cells = []
    for i in range(num_ele_x):
        for j in range(num_ele_y):
            cells.append(
                [
                    i * (num_ele_y + 1) + j,
                    (i + 1) * (num_ele_y + 1) + j,
                    (i + 1) * (num_ele_y + 1) + j + 1,
                    i * (num_ele_y + 1) + j + 1,
                ]
            )
    cells = np.array(cells)

    # boundary cells
    traction_cells = []
    for cell in cells:
        for face in get_faces("quad", cell):
            if np.all(np.abs(points[face] @ np.array([1, 0]) - length) < 1e-20):
                traction_cells.append(face)

    mesh = meshio.Mesh(points, {"quad": cells, "line": np.array(traction_cells)})

    return mesh
