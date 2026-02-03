import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any


def matrix_to_lammps_triclinic(bounds_matrix: np.ndarray) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Convert a 3x3 box matrix to LAMMPS triclinic format (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz).
    
    LAMMPS uses a lower triangular matrix convention:
    [ a_x   0     0   ]
    [ b_x   b_y   0   ]
    [ c_x   c_y   c_z ]
    
    This function converts an arbitrary 3x3 matrix (where rows are lattice vectors) to LAMMPS format
    using Gram-Schmidt orthogonalization.
    
    Parameters:
        bounds_matrix: 3x3 array where each row is a lattice vector
        
    Returns:
        Tuple of (xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz)
    """
    if bounds_matrix.shape != (3, 3):
        raise ValueError("bounds_matrix must be a 3x3 array")
    
    # Extract lattice vectors (rows of the matrix)
    a = bounds_matrix[0].copy()
    b = bounds_matrix[1].copy()
    c = bounds_matrix[2].copy()
    
    # Check if already in LAMMPS format (lower triangular)
    if (abs(a[1]) < 1e-10 and abs(a[2]) < 1e-10 and 
        abs(b[2]) < 1e-10):
        # Already in LAMMPS format
        a_x = a[0]
        b_x, b_y = b[0], b[1]
        c_x, c_y, c_z = c[0], c[1], c[2]
    else:
        # Convert to LAMMPS format using Gram-Schmidt process
        # a vector along x-axis
        a_x = np.linalg.norm(a)
        if a_x < 1e-10:
            raise ValueError("First lattice vector has zero length")
        a_unit = a / a_x
        
        # b vector: project onto a, then get perpendicular component
        b_x = np.dot(b, a_unit)
        b_perp = b - b_x * a_unit
        b_y = np.linalg.norm(b_perp)
        
        # c vector: project onto a and b, then get perpendicular component
        c_x = np.dot(c, a_unit)
        if b_y > 1e-10:
            b_unit = b_perp / b_y
            c_y = np.dot(c, b_unit)
            c_perp = c - c_x * a_unit - c_y * b_unit
        else:
            c_y = 0.0
            c_perp = c - c_x * a_unit
        c_z = np.linalg.norm(c_perp)
    
    xlo, xhi = 0.0, a_x
    ylo, yhi = 0.0, b_y
    zlo, zhi = 0.0, c_z
    xy, xz, yz = b_x, c_x, c_y
    
    return xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz


def is_orthogonal(bounds_matrix: np.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if a box matrix represents an orthogonal box.
    
    Parameters:
        bounds_matrix: 3x3 array
        tol: Tolerance for checking orthogonality
        
    Returns:
        True if the box is orthogonal (within tolerance)
    """
    if bounds_matrix.shape != (3, 3):
        return False
    
    # Check if off-diagonal elements are close to zero
    off_diag = [
        bounds_matrix[0, 1], bounds_matrix[0, 2],
        bounds_matrix[1, 0], bounds_matrix[1, 2],
        bounds_matrix[2, 0], bounds_matrix[2, 1]
    ]
    return all(abs(x) < tol for x in off_diag)


class write_box:
    """
    Write atomic coordinates to various file formats. Supports arbitrary number of properties per atom.

    Parameters:
        filename: Output file path
        atoms_types: List of atom type labels (e.g. ['H','He'])
        num_atoms: Number of atoms for each type (e.g. [20,30])
        lattice_constant: Box dimensions array (3,), optional
        coordinates: Atomic data array (N_atoms, N_properties), optional
                    Can include xyz coordinates, velocities, weights, etc.
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
        ele_name_idx: Optional[Dict[str, int]] = None,
        bounds_matrix: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize write_box class.
        
        Parameters:
            filename: Output file path
            atoms_types: List of atom type labels (e.g. ['H','He'])
            num_atoms: Number of atoms for each type (e.g. [20,30])
            lattice_constant: Box dimensions array (3,) for orthogonal box, optional
            coordinates: Atomic data array (N_atoms, N_properties), optional
            ele_name_idx: Element type mapping {'element A': 1, ...}, optional
            bounds_matrix: Box matrix (3,3) for triclinic box, optional
                           If provided, takes precedence over lattice_constant
        """
        self.filename = filename
        self.atoms_types = atoms_types
        self.num_atoms = num_atoms
        self.lattice_constant = lattice_constant
        self.coordinates = coordinates
        self.ele_name_idx = ele_name_idx
        self.bounds_matrix = bounds_matrix
        
        # If bounds_matrix is provided, check if it's orthogonal and extract lattice_constant
        if bounds_matrix is not None:
            if bounds_matrix.shape != (3, 3):
                raise ValueError("bounds_matrix must be a 3x3 array")
            if is_orthogonal(bounds_matrix):
                # Extract diagonal elements for orthogonal box
                self.lattice_constant = np.diag(bounds_matrix)

    def write_vasp_poscar_file(self) -> None:
        """
        Write a VASP POSCAR file, single frame.
        Supports both orthogonal and triclinic boxes.
        """
        ret = ""
        ret += "whatever\n"
        ret += "1.0\n"
        
        # Use bounds_matrix if available, otherwise use lattice_constant
        if self.bounds_matrix is not None:
            # Write matrix directly (each row is a lattice vector)
            for i in range(3):
                ret += f"{self.bounds_matrix[i, 0]:.16f} {self.bounds_matrix[i, 1]:.16f} {self.bounds_matrix[i, 2]:.16f}\n"
        elif self.lattice_constant is not None:
            # Orthogonal box
            ret += f"{self.lattice_constant[0]} 0.0 0.0\n"
            ret += f"0.0 {self.lattice_constant[1]} 0.0\n"
            ret += f"0.0 0.0 {self.lattice_constant[2]}\n"
        else:
            raise ValueError("Either lattice_constant or bounds_matrix must be provided")
        
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

    def write_lammps_data_file(self) -> None:
        """
        Write a LAMMPS data file, single frame.
        Supports both orthogonal and triclinic boxes.
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
        
        # Handle box bounds
        if self.bounds_matrix is not None:
            # Check if it's orthogonal
            if is_orthogonal(self.bounds_matrix):
                # Orthogonal box
                xlo, xhi = 0.0, self.bounds_matrix[0, 0]
                ylo, yhi = 0.0, self.bounds_matrix[1, 1]
                zlo, zhi = 0.0, self.bounds_matrix[2, 2]
                ret += f"{xlo:.16f} {xhi:.16f} xlo xhi\n"
                ret += f"{ylo:.16f} {yhi:.16f} ylo yhi\n"
                ret += f"{zlo:.16f} {zhi:.16f} zlo zhi\n\n"
            else:
                # Triclinic box
                xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = matrix_to_lammps_triclinic(self.bounds_matrix)
                ret += f"{xlo:.16f} {xhi:.16f} xlo xhi\n"
                ret += f"{ylo:.16f} {yhi:.16f} ylo yhi\n"
                ret += f"{zlo:.16f} {zhi:.16f} zlo zhi\n"
                ret += f"{xy:.16f} {xz:.16f} {yz:.16f} xy xz yz\n\n"
        elif self.lattice_constant is not None:
            # Orthogonal box from lattice_constant
            ret += f"0.0 {self.lattice_constant[0]:.16f} xlo xhi\n"
            ret += f"0.0 {self.lattice_constant[1]:.16f} ylo yhi\n"
            ret += f"0.0 {self.lattice_constant[2]:.16f} zlo zhi\n\n"
        else:
            raise ValueError("Either lattice_constant or bounds_matrix must be provided")
        
        ret += "Atoms\n\n"

        atom_id = 1
        for element, number in zip(self.atoms_types, self.num_atoms):
            element_type = self.ele_name_idx[element]
            for _ in range(number):
                atom_data = self.coordinates[atom_id - 1]
                ret += f"{atom_id} {element_type}"
                for value in atom_data:
                    ret += f" {value:.6f}"
                ret += f" # {element}\n"
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
            
            # Check if box is triclinic
            current_matrix = bounds_matrix[step]
            if is_orthogonal(current_matrix):
                # Orthogonal box
                ret += "ITEM: BOX BOUNDS pp pp pp\n"
                cell_length = np.diag(current_matrix)
                ret += f"0.0 {cell_length[0]:.16f}\n"
                ret += f"0.0 {cell_length[1]:.16f}\n"
                ret += f"0.0 {cell_length[2]:.16f}\n"
            else:
                # Triclinic box
                ret += "ITEM: BOX BOUNDS pp pp pp xy xz yz\n"
                xlo, xhi, ylo, yhi, zlo, zhi, xy, xz, yz = matrix_to_lammps_triclinic(current_matrix)
                ret += f"{xlo:.16f} {xhi:.16f} {xy:.16f}\n"
                ret += f"{ylo:.16f} {yhi:.16f} {xz:.16f}\n"
                ret += f"{zlo:.16f} {zhi:.16f} {yz:.16f}\n"

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
                        data_line += f" {value:.6f}"
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
            coordinates: Atomic data array (Nframes, N_atoms, N_properties)
                        Can include xyz coordinates, velocities, weights, etc.
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
                    atom_data = coordinates[frame_idx, atom_id - 1]
                    ret += f"{element}"
                    for value in atom_data:
                        ret += f" {value:.6f}"
                    ret += "\n"
                    atom_id += 1

        with open(self.filename, "w") as f:
            f.write(ret)