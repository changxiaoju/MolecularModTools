from typing import List, Tuple, Optional, Dict
import numpy as np


def write_dimer(
    filename: str,
    coordinates: np.ndarray,
    atom_type: List[str],
    cell_lengths: Optional[np.ndarray] = None,
    comment: str = ""
) -> None:
    """
    Write dimer configuration to file.

    Parameters:
        filename: Output file path
        coordinates: Atomic coordinates
        atom_type: List of atom type labels
        cell_lengths: Box dimensions (optional)
        comment: Comment line for the file (optional)
    """
    with open(filename, "w") as dimer_file:
        dimer_file.write(f"{len(coordinates)}\n")
        dimer_file.write(comment + "\n")
        
        if cell_lengths is not None:
            for length in cell_lengths:
                dimer_file.write(f"{length:.6f} ")
            dimer_file.write("\n")
            
        for i, coord in enumerate(coordinates):
            dimer_file.write(f"{atom_type[i]} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

class WriteDimer:
    """
    Output absolute coordinates file only
    Orthogonal box only
    Elements, numbers and element_types should be in line with coordinates
    lattice_constant: a constant
    coordinates: [(x,y,z,dx,dy,dz),...], dx,dy,dz are for shift or bond-pair adjustment
    elements: ['element A','element B',....]
    numbers: ['number of element A','number of element B',...]
    element_types: ['element A': 1, 'element B': 2,...]

    """

    def __init__(
        self,
        lattice_constant: float,
        coordinates: List[Tuple[float, float, float, float, float, float]],
        elements: List[str],
        numbers: List[int],
        element_types: Dict[str, int]
    ) -> None:
        self.lattice_constant = lattice_constant
        self.coordinates = coordinates
        self.elements = elements
        self.numbers = numbers
        self.element_types = element_types

    def generate_lammps_data_file(self, filename: str) -> None:
        ret = ""
        ret += f"LAMMPS data file for "
        for element, number in zip(self.elements, self.numbers):
            ret += f"{number} {element} "
        ret = ret.rstrip()  # 移除末尾的空格
        ret += "\n\n"

        total_atoms = sum(self.numbers)
        num_unique_elements = len(set(self.elements))
        ret += f"{total_atoms} atoms\n"
        ret += f"{num_unique_elements} atom types\n\n"
        ret += f"0.0 {self.lattice_constant} xlo xhi\n"
        ret += f"0.0 {self.lattice_constant} ylo yhi\n"
        ret += f"0.0 {self.lattice_constant} zlo zhi\n\n"
        ret += "Atoms\n\n"

        atom_id = 1
        for element, number in zip(self.elements, self.numbers):
            element_type = self.element_types[element]
            for _ in range(number):
                x, y, z, hx, hy, hz = self.coordinates[atom_id - 1]
                ret += f"{atom_id} {element_type} {x+hx:.4f} {y+hy:.4f} {z+hz:.4f} # {element}\n"
                atom_id += 1

        with open(filename, "w") as f:
            f.write(ret)

    def generate_vasp_poscar_file(self, filename: str) -> None:
        ret = ""
        ret += "whatever\n"
        ret += "1.0\n"
        ret += f"{self.lattice_constant} 0.0 0.0\n"
        ret += f"0.0 {self.lattice_constant} 0.0\n"
        ret += f"0.0 0.0 {self.lattice_constant}\n"
        ret += " ".join(self.elements) + "\n"
        ret += " ".join(map(str, self.numbers)) + "\n"
        ret += "Cartesian\n"

        atom_id = 1
        for element, number in zip(self.elements, self.numbers):
            for _ in range(number):
                x, y, z, hx, hy, hz = self.coordinates[atom_id - 1]
                ret += f"{x+hx:.6f} {y+hy:.6f} {z+hz:.6f}\n"
                atom_id += 1

        with open(filename, "w") as f:
            f.write(ret)
