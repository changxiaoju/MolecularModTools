import numpy as np

class SuperCell:
    def __init__(self, lattice, positions):
        """
        Create a SuperCell object.

        Parameters:
        lattice (float): The lattice lenth of the unit cell.
        positions (np.array): The atomic positions in the unit cell.
        """
        self.lattice = lattice
        self.positions = positions

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
        new_lattice = np.dot(np.diag(scaling_factors), self.lattice)

        # Generate new atomic positions in the supercell
        new_positions = []
        for i in range(scaling_factors[0]):
            for j in range(scaling_factors[1]):
                for k in range(scaling_factors[2]):
                    translation_vector = np.array([i, j, k])
                    translated_positions = self.positions + np.dot(translation_vector,self.lattice)
                    new_positions.extend(translated_positions)

        return SuperCell(new_lattice, np.array(new_positions))

