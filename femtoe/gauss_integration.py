from typing import List, Tuple
import itertools
from typing import Iterator
import numpy as np
import math


def gauss_integration(cell_type: str, numgp: int) -> List[Tuple[float, np.ndarray]]:
    if cell_type == "quad" or cell_type == "quad8" or cell_type == "quad9":

        if numgp == 1:
            return [(4.0, np.array([0.0, 0.0]))]
        elif numgp == 4:
            return [
                (1.0, np.array([-1.0 / math.sqrt(3), -1.0 / math.sqrt(3)])),
                (1.0, np.array([1.0 / math.sqrt(3), -1.0 / math.sqrt(3)])),
                (1.0, np.array([1.0 / math.sqrt(3), 1.0 / math.sqrt(3)])),
                (1.0, np.array([-1.0 / math.sqrt(3), 1.0 / math.sqrt(3)])),
            ]

        raise NotImplementedError(f"Unsupported number of Gauss pounts for {cell_type}")

    if cell_type == "line" or cell_type == "line3":

        if numgp == 1:
            return [(2.0, np.array(0.0))]
        elif numgp == 2:
            return [
                (1.0, np.array(-1.0 / math.sqrt(3))),
                (1.0, np.array(1.0 / math.sqrt(3))),
            ]

    raise NotImplementedError(f"Unsupported cell type {cell_type}")
