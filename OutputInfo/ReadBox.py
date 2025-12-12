import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union


def read_xdatcar_frame(
    filename: str,
    interval: int = 1
) -> Tuple[List[str], Union[Tuple[int, List[Dict], np.ndarray], List[Tuple[int, List[Dict], np.ndarray]]], List[str]]:
    """
    Read atomic coordinates from VASP XDATCAR file.

    Parameters:
        filename: Path to XDATCAR file
        interval: Reading interval for frames

    Returns:
        Tuple containing:
            - List of frame data (timestep, coordinates) or a single array (coordinates)
            - Cell Vectors
            - List of atom numbers
    """
    frames = []

    with open(filename, "r") as XDATCAR_file:
        system_info = XDATCAR_file.readline()
        scale_factor = int(float(XDATCAR_file.readline().strip()))
        cell_vectors = []
        for _ in range(3):
            cell_vectors.append([scale_factor * float(x) for x in XDATCAR_file.readline().split()])
        cell_vectors = np.array(cell_vectors)
        atoms_types = XDATCAR_file.readline().split()[0]
        num_atom = [int(x) for x in XDATCAR_file.readline().split()]

        while True:
            try:
                info_line = XDATCAR_file.readline().split()
                if len(info_line) > 1:
                    coord_type, step = info_line[0], int(info_line[2])
                else:
                    coord_type, step = info_line[0], 0

                if step % interval != 0:
                    for _ in range(np.array(num_atom).sum()):
                        XDATCAR_file.readline()
                    continue

                coordinate = []
                for _ in range(np.array(num_atom).sum()):
                    atom_info = np.array([float(x) for x in XDATCAR_file.readline().split()])
                    if coord_type == "Direct":
                        atom_info %= 1
                        atom_info = atom_info @ cell_vectors
                    else:
                        fractional = np.linalg.solve(cell_vectors.T, atom_info) % 1
                        atom_info = fractional @ cell_vectors
                    coordinate.append(atom_info)

                frames.append((step, np.array(coordinate)))

            except:
                print(f"reading frame step {step}")
                break

    if len(frames) == 1:
        return frames[0][1], cell_vectors, num_atom
    else:
        return frames, cell_vectors, num_atom


def read_xyz_frame(filename: str) -> Tuple[Union[np.ndarray, List[np.ndarray]], List[float], List[str]]:
    """
    Read atomic coordinates from xyz file.

    Parameters:
        filename: Path to xyz file

    Returns:
        Tuple containing:
            - Coordinates (single frame or list of frames)
            - Cell lengths 
            - Atom type labels
    """

    with open(filename, "r") as xyz_file:
        coordinates = []
        cell_lengths = []

        while True:
            try:
                num_atom = int(xyz_file.readline())
                try:
                    cell_length = [float(i) for i in xyz_file.readline().split()]
                except:
                    cell_length = None

                cell_lengths.append(cell_length)
                coordinate = []
                atom_type = []

                for _ in range(num_atom):
                    atom_info = xyz_file.readline().split()
                    atom_type.append(atom_info[0])
                    coordinate.append([float(coord) for coord in atom_info[1:]])

                coordinates.append(np.array(coordinate))

            except:
                print(f"reading frame step {len(coordinates)}")
                break
    if len(coordinates) == 1:
        return coordinates[0], cell_length, atom_type
    else:
        return coordinates, cell_lengths, atom_type

def read_lammps_data_file(
    filename: str
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, List[int]]:
    """
    Read a single LAMMPS data file (write_data output).

    Parameters:
        filename: Path to the LAMMPS data file.

    Returns:
        Tuple containing:
            - Coordinates sorted by atom id (N x 3)
            - Velocities sorted by atom id (N x 3) or None if missing
            - Bounds matrix describing the simulation cell (3 x 3)
            - Atom type list sorted by atom id
    """
    atoms_section: List[str] = []
    velocities_section: List[str] = []
    current_section: Optional[str] = None

    num_atoms: Optional[int] = None
    num_atom_types: Optional[int] = None
    bounds = {"xlo": 0.0, "xhi": 0.0, "ylo": 0.0, "yhi": 0.0, "zlo": 0.0, "zhi": 0.0}
    tilts = {"xy": 0.0, "xz": 0.0, "yz": 0.0}

    known_headers = {
        "masses",
        "atoms",
        "velocities",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "pair",
        "bond",
        "angle",
        "dihedral",
        "improper",
    }

    with open(filename, "r") as data_file:
        for raw_line in data_file:
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            lower_first = tokens[0].lower()

            # Handle global counts and box bounds located in the header
            if len(tokens) == 2 and tokens[1].lower() == "atoms":
                num_atoms = int(tokens[0])
                continue
            if len(tokens) == 3 and tokens[2].lower() == "atom" and tokens[1].lower() == "types":
                num_atom_types = int(tokens[0])
                continue
            if len(tokens) >= 4 and tokens[-2:] == ["xlo", "xhi"]:
                bounds["xlo"], bounds["xhi"] = float(tokens[0]), float(tokens[1])
                continue
            if len(tokens) >= 4 and tokens[-2:] == ["ylo", "yhi"]:
                bounds["ylo"], bounds["yhi"] = float(tokens[0]), float(tokens[1])
                continue
            if len(tokens) >= 4 and tokens[-2:] == ["zlo", "zhi"]:
                bounds["zlo"], bounds["zhi"] = float(tokens[0]), float(tokens[1])
                continue
            if len(tokens) >= 6 and tokens[-3:] == ["xy", "xz", "yz"]:
                tilts["xy"], tilts["xz"], tilts["yz"] = map(float, tokens[:3])
                continue

            # Switch sections whenever a known header appears
            if lower_first in known_headers:
                if lower_first == "atoms":
                    current_section = "atoms"
                elif lower_first == "velocities":
                    current_section = "velocities"
                else:
                    current_section = None
                continue

            if current_section == "atoms":
                atoms_section.append(line)
            elif current_section == "velocities":
                velocities_section.append(line)

    if num_atoms is None or not atoms_section:
        raise ValueError(f"Could not find atom data in {filename}")

    atom_records = []
    for entry in atoms_section:
        parts = entry.split()
        if len(parts) < 5:
            raise ValueError(f"Atom line has insufficient columns: {entry}")
        atom_id = int(parts[0])
        atom_type = int(parts[1])
        x, y, z = map(float, parts[2:5])
        atom_records.append((atom_id, atom_type, x, y, z))

    atom_records.sort(key=lambda rec: rec[0])
    coordinates = np.array([[rec[2], rec[3], rec[4]] for rec in atom_records])
    atom_types = [rec[1] for rec in atom_records]

    velocities_array: Optional[np.ndarray] = None
    if velocities_section:
        velocity_records = []
        for entry in velocities_section:
            parts = entry.split()
            if len(parts) < 4:
                raise ValueError(f"Velocity line has insufficient columns: {entry}")
            atom_id = int(parts[0])
            vx, vy, vz = map(float, parts[1:4])
            velocity_records.append((atom_id, vx, vy, vz))
        velocity_records.sort(key=lambda rec: rec[0])
        velocities_array = np.array([[rec[1], rec[2], rec[3]] for rec in velocity_records])

    bounds_matrix = np.array(
        [
            [bounds["xhi"] - bounds["xlo"], tilts["xy"], tilts["xz"]],
            [0.0, bounds["yhi"] - bounds["ylo"], tilts["yz"]],
            [0.0, 0.0, bounds["zhi"] - bounds["zlo"]],
        ]
    )

    return coordinates, velocities_array, bounds_matrix, atom_types


def read_lammpstrj_dump(filename: str, interval: int = 1) -> Tuple[List[Tuple[int, np.ndarray, np.ndarray]], List[int]]:
    """
    Read lammpstrj file with a specified interval.
    lammpstrj file contains only "id type x y z" or "id type xs ys zs"

    Parameters:
        filename: Path to lammpstrj file
        interval: Reading interval for frames

    Returns:
        Tuple containing:
            - List of frame data (timestep, coordinates, box bounds) or a single tuple (timestep, coordinates, box bounds)
            - Global atom types
    """

    frames = []
    atom_type_global = None  # Initialize outside loop

    with open(filename, "r") as lammpstrj_dump_file:

        steps = []
        while True:
            line = lammpstrj_dump_file.readline()
            if not line:
                break  # End of file reached

            if "ITEM: TIMESTEP" in line:
                step = int(lammpstrj_dump_file.readline())
                steps.append(step)
                lammpstrj_dump_file.readline()  # Skip ITEM: NUMBER OF ATOMS line

                if (len(steps) - 1) % interval != 0:  # Subtract one is to consider the zeroth step
                    nn = int(lammpstrj_dump_file.readline())
                    for _ in range(5 + nn):  # Skip the rest of this timestep
                        lammpstrj_dump_file.readline()
                    continue

                num_atom = int(lammpstrj_dump_file.readline())
                box_bounds_line = lammpstrj_dump_file.readline()
                is_triclinic = "xy" in box_bounds_line

                if is_triclinic:
                    min_bound_x, max_bound_x, skew_xy = [float(i) for i in lammpstrj_dump_file.readline().split()]
                    min_bound_y, max_bound_y, skew_xz = [float(i) for i in lammpstrj_dump_file.readline().split()]
                    min_bound_z, max_bound_z, skew_yz = [float(i) for i in lammpstrj_dump_file.readline().split()]

                    bounds_matrix = np.array(
                        [
                            [max_bound_x - min_bound_x, skew_xy, skew_xz],
                            [0, max_bound_y - min_bound_y, skew_yz],
                            [0, 0, max_bound_z - min_bound_z],
                        ]
                    )
                else:
                    min_bound_x, max_bound_x = [float(i) for i in lammpstrj_dump_file.readline().split()]
                    min_bound_y, max_bound_y = [float(i) for i in lammpstrj_dump_file.readline().split()]
                    min_bound_z, max_bound_z = [float(i) for i in lammpstrj_dump_file.readline().split()]

                    bounds_matrix = np.array(
                        [
                            [max_bound_x - min_bound_x, 0, 0],
                            [0, max_bound_y - min_bound_y, 0],
                            [0, 0, max_bound_z - min_bound_z],
                        ]
                    )

                info_line = lammpstrj_dump_file.readline().split()

                if any(keyword in info_line for keyword in ["xs", "ys", "zs"]):
                    coordtype = "relative"
                else:
                    coordtype = "absolute"

                atom_data = []
                for _ in range(num_atom):
                    atom_info = lammpstrj_dump_file.readline().split()
                    atom_id = int(atom_info[0])
                    atom_type = int(atom_info[1])
                    x, y, z = [float(coord) for coord in atom_info[2:5]]

                    if coordtype == "relative":
                        x = x * bounds_matrix[0, 0] + min_bound_x
                        y = y * bounds_matrix[1, 1] + min_bound_y
                        z = z * bounds_matrix[2, 2] + min_bound_z

                    wrap_x = (x - min_bound_x) % bounds_matrix[0, 0]
                    wrap_y = (y - min_bound_y) % bounds_matrix[1, 1]
                    wrap_z = (z - min_bound_z) % bounds_matrix[2, 2]

                    atom_data.append((atom_id, atom_type, wrap_x, wrap_y, wrap_z))

                # Sorting by atom index
                atom_data_sorted = sorted(atom_data, key=lambda x: x[0])
                coordinate_sorted = np.array([[data[2], data[3], data[4]] for data in atom_data_sorted])

                if atom_type_global is None:
                    atom_type_global = [data[1] for data in atom_data_sorted]  # Extract atom types once

                frames.append((step, coordinate_sorted, bounds_matrix))

    if len(frames) == 1:
        return frames[0], atom_type_global
    else:
        return frames, atom_type_global


def read_lammps_dump(
    filename: str,
    interval: int = 1
) -> Tuple[List[str], List[Tuple[int, List[Dict[str, float]], np.ndarray]], List[str]]:
    """
    Read lammps dump file with a specified interval.

    Parameters:
        filename: Path to dump file
        interval: Reading interval for frames

    Returns:
        Tuple containing:
            - Header information
            - List of frames (timestep, atom data, box bounds) or a single tuple (timestep, atom data, box bounds)
            - Global atom types
    """
    frames = []
    atom_type_global = None  # Initialize outside loop

    with open(filename, "r") as lammps_dump_file:

        steps = []
        while True:

            line = lammps_dump_file.readline()
            if not line:
                break  # End of file reached
            if "ITEM: TIMESTEP" in line:
                step = int(lammps_dump_file.readline())
                steps.append(step)
                lammps_dump_file.readline()  # Skip ITEM: NUMBER OF ATOMS line

                if (len(steps) - 1) % interval != 0:  # Subtract one is to consider the zeroth step
                    nn = int(lammps_dump_file.readline())
                    for _ in range(5 + nn):  # Skip the rest of this timestep
                        lammps_dump_file.readline()
                    continue
                num_atom = int(lammps_dump_file.readline())
                box_bounds_line = lammps_dump_file.readline()
                is_triclinic = "xy" in box_bounds_line

                if is_triclinic:
                    min_bound_x, max_bound_x, skew_xy = [float(i) for i in lammps_dump_file.readline().split()]
                    min_bound_y, max_bound_y, skew_xz = [float(i) for i in lammps_dump_file.readline().split()]
                    min_bound_z, max_bound_z, skew_yz = [float(i) for i in lammps_dump_file.readline().split()]

                    bounds_matrix = np.array(
                        [
                            [max_bound_x - min_bound_x, skew_xy, skew_xz],
                            [0, max_bound_y - min_bound_y, skew_yz],
                            [0, 0, max_bound_z - min_bound_z],
                        ]
                    )
                else:
                    min_bound_x, max_bound_x = [float(i) for i in lammps_dump_file.readline().split()]
                    min_bound_y, max_bound_y = [float(i) for i in lammps_dump_file.readline().split()]
                    min_bound_z, max_bound_z = [float(i) for i in lammps_dump_file.readline().split()]

                    bounds_matrix = np.array(
                        [
                            [max_bound_x - min_bound_x, 0, 0],
                            [0, max_bound_y - min_bound_y, 0],
                            [0, 0, max_bound_z - min_bound_z],
                        ]
                    )

                # Read atom properties from ITEM: ATOMS line
                info_line = (
                    lammps_dump_file.readline().strip().split()[2:]
                )  # Extract property names (e.g., id, type, xs, ys, zs, etc.)

                atom_data = []
                for _ in range(num_atom):
                    atom_info = lammps_dump_file.readline().split()
                    atom_data_dict = {
                        key: float(value) if key not in ["id", "type", "mol"] else int(value)
                        for key, value in zip(info_line, atom_info)
                    }

                    # Convert relative coordinates if applicable
                    if "xs" in info_line:
                        atom_data_dict["x"] = atom_data_dict["xs"] * bounds_matrix[0, 0] + min_bound_x
                    if "ys" in info_line:
                        atom_data_dict["y"] = atom_data_dict["ys"] * bounds_matrix[1, 1] + min_bound_y
                    if "zs" in info_line:
                        atom_data_dict["z"] = atom_data_dict["zs"] * bounds_matrix[2, 2] + min_bound_z

                    # Wrap coordinates within the box for absolute coordinates
                    if "x" in atom_data_dict:
                        atom_data_dict["x"] = (atom_data_dict["x"] - min_bound_x) % bounds_matrix[0, 0]
                    if "y" in atom_data_dict:
                        atom_data_dict["y"] = (atom_data_dict["y"] - min_bound_y) % bounds_matrix[1, 1]
                    if "z" in atom_data_dict:
                        atom_data_dict["z"] = (atom_data_dict["z"] - min_bound_z) % bounds_matrix[2, 2]

                    atom_data.append(atom_data_dict)

                # Sort atom_data by 'id' for consistency
                atom_data_sorted = sorted(atom_data, key=lambda x: x["id"])

                if atom_type_global is None:
                    atom_type_global = [data["type"] for data in atom_data_sorted]

                frames.append((step, atom_data_sorted, bounds_matrix))

    if len(frames) == 1:
        return info_line, frames[0], atom_type_global
    else:
        return info_line, frames, atom_type_global
        
def extract_dump_data(
    dumpinfo_list: List[List[Dict[str, float]]],
    properties: List[str]
) -> np.ndarray:
    """
    Extract specific properties from dump data.

    Parameters:
        dumpinfo_list: List of atom data dictionaries
        properties: List of property names to extract

    Returns:
        np.ndarray: Array of extracted properties
    """
    extracted_data = []

    for dumpinfo in dumpinfo_list:
        frame_data = np.array([[atom[prop] for prop in properties] for atom in dumpinfo])
        extracted_data.append(frame_data)

    return np.array(extracted_data)
