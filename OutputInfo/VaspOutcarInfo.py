import os, re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class VaspOutcarInfo:
    def __init__(self, path: str, NPT: bool = False) -> None:
        """
        Initializes the OutcarAnalyzer with the specified path and NPT flag.
        """
        self.path = path
        self.NPT = NPT
        self.time_step: Optional[float] = None
        self.sigma: Optional[float] = None
        self.element_mass: Dict[str, float] = {}
        self.element_count: Dict[str, int] = {}
        self.volume: Optional[float] = None
        self.volumes: List[float] = []
        self.density: Optional[float] = None
        self.densities: List[float] = []

    def basic_info(self) -> Optional[Tuple[float, float, Dict[str, float], Dict[str, int], float, float]]:
        """
        Reads basic information such as time step, element masses, sigma, and volume from the OUTCAR file.
        If NPT is False, it also calculates density based on a single volume.
        """
        outcar_file = os.path.join(self.path, "OUTCAR")

        if not os.path.exists(outcar_file):
            print("OUTCAR file not found.")
            return None

        foundsigma, foundelemass, foundelecount, foundtimestep, foundvolume = False, False, False, False, False

        with open(outcar_file) as outcar:

            if self.NPT:
                foundvolume = True

            while not (foundsigma and foundelemass and foundelecount and foundtimestep and foundvolume):
                line = outcar.readline()

                if not line:
                    break

                if self.time_step is None and "POTIM" in line:
                    match = re.search(r"=\s*([\d.]+)", line)
                    self.time_step = float(match.group(1))
                    foundtimestep = True

                if self.sigma is None and "Fermi-smearing in eV        SIGMA" in line:
                    match = re.search(r"=\s*([\d.]+)", line)
                    self.sigma = float(match.group(1))
                    foundsigma = True

                elif "POMASS" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        masses = parts[1].split()
                        for i in range(len(masses)):
                            mass = float(masses[i])
                            self.element_mass[f"type{i}"] = mass
                    foundelemass = True

                elif "ions per type" in line:
                    parts = line.split("=")
                    counts = parts[1].split()
                    for i in range(len(counts)):
                        count = int(counts[i])
                        self.element_count[f"type{i}"] = count
                    foundelecount = True

                elif not self.NPT and "volume of cell" in line and not foundvolume:
                    self.volume = float(line.split()[4])
                    foundvolume = True

            if not self.NPT:
                self.density = self.calculate_density(self.volume)
                return self.time_step, self.sigma, self.element_mass, self.element_count, self.volume, self.density
            else:
                return self.time_step, self.sigma, self.element_mass, self.element_count

    def thermo_info(self) -> Optional[Tuple[List[float], List[float], List[float], List[np.ndarray], List[np.ndarray]]]:
        """
        Reads thermodynamic information such as temperature, volume, energy, pressure,
        and optionally atomic positions and forces from the OUTCAR file.
        If NPT is True, it calculates density based on varying volumes.
        """
        outcar_file = os.path.join(self.path, "OUTCAR")
        temperature = []
        pressure = []
        energy = []
        positions = []
        forces = []

        if not os.path.exists(outcar_file):
            print("OUTCAR file not found.")
            return None

        with open(outcar_file, "r") as outcar:
            reading_position_force = False
            temp_positions, temp_forces = [], []

            for line in outcar:
                if "(temperature" in line:
                    temperature.append(float(line.split()[5]))
                elif self.NPT and "volume of cell" in line:
                    vol = float(line.split()[4])
                    self.volumes.append(vol)
                elif "free  energy   TOTEN" in line:
                    energy.append(float(line.split()[4]))
                elif "total pressure" in line:
                    pressure.append(float(line.split()[3]))
                elif "TOTAL-FORCE" in line:
                    reading_position_force = True
                    next(outcar)  # Skip separator line
                    continue
                if reading_position_force:
                    if "------" in line:
                        positions.append(np.array(temp_positions))
                        forces.append(np.array(temp_forces))
                        temp_positions, temp_forces = [], []
                        reading_position_force = False
                    else:
                        parts = line.split()
                        position = [float(x) for x in parts[:3]]
                        force = [float(x) for x in parts[3:6]]
                        temp_positions.append(position)
                        temp_forces.append(force)

        if self.NPT:
            # i have no idea how NPT ouput style will be
            self.volumes = self.volumes[2:]
            self.densities = [self.calculate_density(vol) for vol in self.volumes]
            return temperature, pressure, energy, self.volumes, self.densities, positions, forces
        else:
            return temperature, pressure, energy, positions, forces

    def calculate_density(self, volume: float) -> float:
        """
        Calculate the density of the system in g/cm³.
        """
        total_mass = sum(self.element_mass[element] * self.element_count[element] for element in self.element_mass)
        atomic_mass_unit_to_gram = 1.66053906660e-24  # Conversion factor from amu to grams
        density = (total_mass * atomic_mass_unit_to_gram) / (volume * 1e-24)  # Convert Å³ to cm³

        return density
