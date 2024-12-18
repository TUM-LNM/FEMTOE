import numpy as np


def local_to_global(cell, num_dof_per_ele):
    return np.vstack([2 * cell + i for i in range(num_dof_per_ele)]).T.flatten()


def assemble_matrix(global_matrix, element_matrix, cell, num_dof_per_ele):
    index_mapping = local_to_global(cell, num_dof_per_ele)

    # assemble matrix
    global_matrix[np.ix_(index_mapping, index_mapping)] += element_matrix


def assemble_vector(global_vector, element_vector, cell, num_dof_per_ele):
    index_mapping = local_to_global(cell, num_dof_per_ele)

    # assemble matrix
    global_vector[index_mapping] -= element_vector
