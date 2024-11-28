class write_box:
    """
    # Output absolute coordinates file only

    Parameters:
        output_filename: str
            The name of the output file.
        atoms_types: list of strs
            E.g.,['H','He'].
        num_atoms: list of numbers
            E.g.,[20,30] for 20 H atoms ans 30 He atoms.
        lattice_constant: np.adrray, optional
            For single frame.
            A one-dimensional array of floats with shape (3,).
        coordinates: np.ndarray, optional
            For single frame.
            A two-dimensional array of floats with shape (number of atoms, 3).
        ele_name_idx: dict, optional
            For lammps file.
            {'element A': 1, 'element B': 2,...}
    Returns:
    """

    def __init__(self, filename, atoms_types, num_atoms, lattice_constant=None, coordinates=None, ele_name_idx=None):
        self.filename = filename
        self.atoms_types = atoms_types
        self.num_atoms = num_atoms
        self.lattice_constant = lattice_constant
        self.coordinates = coordinates
        self.ele_name_idx = ele_name_idx

    def write_lammps_data_file(self):
        """
        Write a LAMMPS data file, single frame.
        """
        ret = ""
        ret += f"LAMMPS data file for "
        for element, number in zip(self.atoms_types, self.num_atoms):
            ret += f"{number} {element} "
        ret = ret.rstrip()  # 移除末尾的空格
        ret += "\n\n"

        num_unique_elements = len(set(self.atoms_types))
        ret += f"{sum(self.num_atoms)} atoms\n"
        ret += f"{num_unique_elements} atom types\n\n"
        ret += f"0.0 {self.lattice_constant[0]} xlo xhi\n"
        ret += f"0.0 {self.lattice_constant[1]} ylo yhi\n"
        ret += f"0.0 {self.lattice_constant[2]} zlo zhi\n\n"
        ret += "Atoms\n\n"

        atom_id = 1
        for element, number in zip(self.atoms_types, self.num_atoms):
            element_type = self.ele_name_idx[element]
            for _ in range(number):
                x, y, z = self.coordinates[atom_id - 1]
                ret += f"{atom_id} {element_type} {x:.4f} {y:.4f} {z:.4f} # {element}\n"
                atom_id += 1

        with open(self.filename, "w") as f:
            f.write(ret)

    def write_vasp_poscar_file(self):
        """
        Write a VASP POSCAR file, single frame.
        """

        ret = ""
        ret += "whatever\n"
        ret += "1.0\n"
        ret += f"{self.lattice_constant[0]} 0.0 0.0\n"
        ret += f"0.0 {self.lattice_constant[1]} 0.0\n"
        ret += f"0.0 0.0 {self.lattice_constant[2]}\n"
        ret += " ".join(self.atoms_types) + "\n"
        ret += " ".join(map(str, self.num_atoms)) + "\n"
        ret += "Cartesian\n"

        atom_id = 1
        for element, number in zip(self.atoms_types, self.num_atoms):
            for _ in range(number):
                x, y, z = self.coordinates[atom_id - 1]
                ret += f"{x:.6f} {y:.6f} {z:.6f}\n"
                atom_id += 1

        with open(self.filename, "w") as f:
            f.write(ret)

    def write_lammps_trj_file(self, frames):
        """
        Write a LAMMPS trajectory (.lammpstrj) file with multiple frames.

        Parameters:
            frames: list of tuples
                every tuple contains two numpy arrays.
                - np.ndarray: A two-dimensional array of floats with shape (sum(num_atoms), 3), coordinates.
                - np.ndarray: A one-dimensional array of floats with shape (3,), cell_length.
        """

        ret = ""

        for step, frame in enumerate(frames):

            coordinates, cell_length = frame[0], frame[1]

            ret += f"ITEM: TIMESTEP\n{step}\n"
            ret += "ITEM: NUMBER OF ATOMS\n"
            ret += f"{sum(self.num_atoms)}\n"
            ret += "ITEM: BOX BOUNDS pp pp pp\n"
            ret += f"0.0 {cell_length[0]}\n"
            ret += f"0.0 {cell_length[1]}\n"
            ret += f"0.0 {cell_length[2]}\n"
            ret += "ITEM: ATOMS id type x y z\n"

            atom_id = 1
            for element, number in zip(self.atoms_types, self.num_atoms):
                element_type = self.ele_name_idx[element]
                for _ in range(number):
                    x, y, z = coordinates[atom_id - 1]
                    ret += f"{atom_id} {element_type} {x:.4f} {y:.4f} {z:.4f} # {element}\n"
                    atom_id += 1

        with open(self.filename, "w") as f:
            f.write(ret)
