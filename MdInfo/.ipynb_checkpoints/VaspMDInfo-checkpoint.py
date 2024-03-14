import os,re
import numpy as np

class VaspMDInfo:
    """
    For VASP DFT-MD OUTCAR only
    unit: temperatures (K); sigam (Ry);volumes (A^3); pressures (kB); steps (fs); density (gcc)

    to be add:
    energy = float(re.search(r'free  energy   TOTEN  =\s+-?(\d+\.\d+) eV', content).group(1))
    """
    
    def __init__(self, path):
        self.path = path
        self.temperatures = []
        self.volumes = []
        self.pressures = []
        self.steps = []
        self.element_masses = {}  # 用于存储元素的POMASS
        self.element_counts = {}  # 用于存储元素的数量
        self.time_step = None  # 将时间步长移动到初始化
        self.sigma = None

    def read_data(self):
        outcar_file = os.path.join(self.path, 'OUTCAR')

        if not  os.path.exists(outcar_file):
            print("OUTCAR file not found.")
            return

        with open(outcar_file, 'r') as outcar:
            for line in outcar:
                if self.time_step is None and "POTIM" in line:
                    match = re.search(r'=\s*([\d.]+)', line)
                    self.time_step = float(match.group(1))
                if self.sigma is None and "Fermi-smearing in eV        SIGMA" in line:
                    match = re.search(r'=\s*([\d.]+)', line)
                    self.sigma = float(match.group(1))
                elif "(temperature" in line:
                    temperature = float(line.split()[5])
                    self.temperatures.append(temperature)
                elif "volume of cell" in line:
                    volume = float(line.split()[4])
                    self.volumes.append(volume)
                elif "total pressure" in line:
                    pressure = float(line.split()[3])
                    self.pressures.append(pressure)
                elif "POMASS" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        masses = parts[1].split()
                        for i in range(len(masses)):
                            mass = float(masses[i])
                            self.element_masses["type"+str(i)] = mass
                elif "ions per type" in line:
                    parts = line.split("=")
                    counts = parts[1].split()
                    for i in range(len(counts)):
                        count = int(counts[i])
                        self.element_counts["type"+str(i)] = count

        num_steps = len(self.temperatures)
        self.steps = np.arange(1, num_steps + 1) * self.time_step

    def calculate_density(self):
        if len(self.volumes) == 0:
            print("No volume data available.")
            return
        
        total_mass = sum([self.element_masses[element] * self.element_counts[element] for element in self.element_masses])
        
        # 原子质量单位的转换因子（1 amu = 1.66053906660e-24 g）
        atomic_mass_unit_to_gram = 1.66053906660e-24
        # Calculate density in g/cm³
        densities = [(total_mass * atomic_mass_unit_to_gram) / (volume * 1e-24) for volume in self.volumes]

        return densities