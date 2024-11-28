import numpy as np


class adjust_cell:
    def __init__(self, lattice_constant, coordinates):
        """
        Create a SuperCell object.

        Parameters:
            lattice_constant: np.adrray
                For single frame.
                A one-dimensional array of floats with shape (3,).
            coordinates: np.ndarray
                For single frame.
                A two-dimensional array of floats with shape (number of atoms, 3).
        """
        self.lattice_constant = lattice_constant
        self.coordinates = coordinates

    def make_supercell(self, scaling_factors):
        """
        Create a supercell of the crystal structure.

        Parameters:
        scaling_factors (tuple): A tuple of three integers specifying the scaling factors
                                 along the three lattice vectors.

        Returns:
        SuperCell: The supercell object.
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
