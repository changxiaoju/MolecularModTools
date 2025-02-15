import numpy as np
from typing import Tuple, Union, List


class adjust_cell:
    def __init__(self, coordinates: np.ndarray, bounds_matrix: np.ndarray, moltype: Union[List[int], np.ndarray]) -> None:
        """
        Create a SuperCell object.

        Parameters:
            coordinates: A two-dimensional array of floats with shape (Natoms, 3)
            bounds_matrix: A two-dimensional array of floats with shape (3, 3)
            moltype : List or numpy array indicating the type of molecules

        """
        self.coordinates = coordinates
        self.bounds_matrix = bounds_matrix
        self.moltype = moltype

    def make_supercell(self, scaling_factors: Union[Tuple[int, int, int], List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a supercell of the crystal structure.

        Parameters:
            scaling_factors: A tuple/list of three integers specifying the scaling factors along the three lattice vectors

        Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - new_lattice: The scaled lattice constants
            - new_coordinates: The coordinates of atoms in the supercell
        """
        scaling_factors = np.array(scaling_factors)
        new_bounds_matrix = scaling_factors * self.bounds_matrix

        # Generate new atomic coordinates in the supercell
        new_coordinates = []
        new_moltype = []
        for i in range(scaling_factors[0]):
            for j in range(scaling_factors[1]):
                for k in range(scaling_factors[2]):
                    translation_vector = np.array([i, j, k])
                    translated_coordinates = self.coordinates + translation_vector * np.diag(self.bounds_matrix)
                    new_coordinates.extend(translated_coordinates)
                    new_moltype.extend(self.moltype)

        return new_bounds_matrix, np.array(new_coordinates), np.array(new_moltype)
