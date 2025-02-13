import numpy as np
from typing import Tuple, Union, List


class adjust_cell:
    def __init__(self, lattice_constant: np.ndarray, coordinates: np.ndarray) -> None:
        """
        Create a SuperCell object.

        Parameters:
            lattice_constant: np.ndarray
                For single frame.
                A one-dimensional array of floats with shape (3,).
            coordinates: np.ndarray
                For single frame.
                A two-dimensional array of floats with shape (Natoms, 3).
        """
        self.lattice_constant = lattice_constant
        self.coordinates = coordinates

    def make_supercell(self, scaling_factors: Union[Tuple[int, int, int], List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a supercell of the crystal structure.

        Parameters:
        scaling_factors: Tuple[int, int, int] or List[int]
            A tuple/list of three integers specifying the scaling factors along the three lattice vectors.

        Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - new_lattice: The scaled lattice constants
            - new_coordinates: The coordinates of atoms in the supercell
        """
        scaling_factors = np.array(scaling_factors)
        new_lattice = scaling_factors * self.lattice_constant

        # Generate new atomic coordinates in the supercell
        new_coordinates = []
        for i in range(scaling_factors[0]):
            for j in range(scaling_factors[1]):
                for k in range(scaling_factors[2]):
                    translation_vector = np.array([i, j, k])
                    translated_coordinates = self.coordinates + translation_vector * self.lattice_constant
                    new_coordinates.extend(translated_coordinates)

        return new_lattice, np.array(new_coordinates)
