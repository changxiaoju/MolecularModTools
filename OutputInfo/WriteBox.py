import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any

class write_box:
    """
    Write atomic coordinates to various file formats. Absolute coordinates only.

    Parameters:
        filename: Output file path
        atoms_types: List of atom type labels (e.g. ['H','He'])
        num_atoms: Number of atoms for each type (e.g. [20,30])
        lattice_constant: Box dimensions array (3,), optional
        coordinates: Atomic coordinates array (N_atoms,3), optional
        ele_name_idx: Element type mapping {'element A': 1, ...}, optional

    Returns:
        None
    """

    def __init__(
        self, 
        filename: str,
        atoms_types: List[str],
        num_atoms: List[int],
        lattice_constant: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None,
        ele_name_idx: Optional[Dict[str, int]] = None
    ) -> None:
        self.filename = filename
        self.atoms_types = atoms_types
        self.num_atoms = num_atoms
        self.lattice_constant = lattice_constant
        self.coordinates = coordinates
        self.ele_name_idx = ele_name_idx

    def write_lammps_data_file(self) -> None:
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

    def write_vasp_poscar_file(self) -> None:
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

    def write_lammps_trj_file(
        self, 
        coordinates: np.ndarray,
        bounds_matrix: np.ndarray,
        property_names: Optional[List[str]] = None
    ) -> None:
        """
        Write a LAMMPS trajectory file with multiple frames.

        Parameters:
            coordinates: Atomic coordinates array (Nframes, N_atoms, 3)
            bounds_matrix: Box matrices array (Nframes, 3,3) or (3,3)
            property_names: Names of properties to dump with coordinates
        """
        if property_names is None:
            property_names = ['x', 'y', 'z']

        # Handle single frame bounds_matrix
        if bounds_matrix.ndim == 2:
            bounds_matrix = np.expand_dims(bounds_matrix, axis=0)
            
        if bounds_matrix.shape[0] != coordinates.shape[0]:
            if bounds_matrix.shape[0] != 1:
                raise ValueError("bounds_matrix must have same number of frames as coordinates or be single frame")
            bounds_matrix = np.repeat(bounds_matrix, coordinates.shape[0], axis=0)

        ret = ""
        for step in range(coordinates.shape[0]):
            ret += f"ITEM: TIMESTEP\n{step}\n"
            ret += "ITEM: NUMBER OF ATOMS\n"
            ret += f"{sum(self.num_atoms)}\n"
            ret += "ITEM: BOX BOUNDS pp pp pp\n"
            cell_length = np.diag(bounds_matrix[step])
            ret += f"0.0 {cell_length[0]}\n"
            ret += f"0.0 {cell_length[1]}\n"
            ret += f"0.0 {cell_length[2]}\n"

            header_line = "ITEM: ATOMS id type"
            for name in property_names:
                header_line += f" {name}"
            header_line += "\n"
            ret += header_line

            atom_id = 1
            for element, number in zip(self.atoms_types, self.num_atoms):
                element_type = self.ele_name_idx[element]
                for _ in range(number):
                    data_line = f"{atom_id} {element_type}"
                    for value in coordinates[step, atom_id - 1]:
                        data_line += f" {value:.4f}"
                    data_line += f" # {element}\n"
                    ret += data_line
                    atom_id += 1

        with open(self.filename, "w") as f:
            f.write(ret)

    def write_xyz_file(
        self, 
        coordinates: np.ndarray,
        bounds_matrix: np.ndarray
    ) -> None:
        """
        Write a xyz trajectory file with multiple frames.

        Parameters:
            coordinates: Atomic coordinates array (Nframes, N_atoms, 3)
            bounds_matrix: Box matrices array (Nframes, 3,3) or (3,3)
        """
        # Handle single frame bounds_matrix
        if bounds_matrix.ndim == 2:
            bounds_matrix = np.expand_dims(bounds_matrix, axis=0)
            
        if bounds_matrix.shape[0] != coordinates.shape[0]:
            if bounds_matrix.shape[0] != 1:
                raise ValueError("bounds_matrix must have same number of frames as coordinates or be single frame")
            bounds_matrix = np.repeat(bounds_matrix, coordinates.shape[0], axis=0)

        ret = ""
        for frame_idx in range(coordinates.shape[0]):
            ret += f"{sum(self.num_atoms)}\n"
            cell_length = np.diag(bounds_matrix[frame_idx])
            ret += f"{cell_length[0]} {cell_length[1]} {cell_length[2]}\n"

            atom_id = 1
            for element, number in zip(self.atoms_types, self.num_atoms):
                for _ in range(number):
                    x, y, z = coordinates[frame_idx, atom_id - 1]
                    ret += f"{element} {x:.4f} {y:.4f} {z:.4f}\n"
                    atom_id += 1

        with open(self.filename, "w") as f:
            f.write(ret)