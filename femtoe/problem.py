import abc
import numpy as np


class Problem(metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def cell_type(self):
        pass

    @property
    @abc.abstractmethod
    def cells(self):
        pass

    @property
    @abc.abstractmethod
    def nodes(self):
        pass

    @property
    @abc.abstractmethod
    def num_dof_per_node(self):
        pass

    @property
    @abc.abstractmethod
    def num_dof_per_element(self):
        pass

    @property
    @abc.abstractmethod
    def num_dof(self):
        pass

    @abc.abstractmethod
    def get_traction(self, x: np.ndarray):
        pass

    @abc.abstractmethod
    def get_body_forces(self, x: np.ndarray):
        pass

    @property
    @abc.abstractmethod
    def dirichlet_dofs(self):
        pass
