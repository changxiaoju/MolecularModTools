import os,re
import numpy as np

class VaspOutcarInfo:
    """
    unit: temperatures (K); sigam (Ry);volume (A^3); pressure (kB); step (fs); density (gcc); energy(eV)

    to be add:
    energy = float(re.search(r'free  energy   TOTEN  =\s+-?(\d+\.\d+) eV', content).group(1))
    """
    
    def __init__(self, path):
        self.path = path
        self.temperature = []
        self.volume = []
        self.pressure = []
        self.steps = []
        self.element_mass = {}  # 用于存储元素的POMASS
        self.element_count = {}  # 用于存储元素的数量
        self.time_step = None  # 将时间步长移动到初始化
        self.sigma = None
        self.energy = []
        self.force = []
        self.position = []
        self.SCF = True # 默认处理scf

    def read_data(self):
        outcar_file = os.path.join(self.path, 'OUTCAR')

        if not  os.path.exists(outcar_file):
            print("OUTCAR file not found.")
            return

        with open(outcar_file, 'r') as outcar:
            
            reading_position_force = False  
            positions,forces = [], []
            
            for line in outcar:
                if self.time_step is None and "POTIM" in line:
                    match = re.search(r'=\s*([\d.]+)', line)
                    self.time_step = float(match.group(1))
                if self.sigma is None and "Fermi-smearing in eV        SIGMA" in line:
                    match = re.search(r'=\s*([\d.]+)', line)
                    self.sigma = float(match.group(1))
                elif "(temperature" in line:
                    temperature = float(line.split()[5])
                    self.temperature.append(temperature)
                elif "volume of cell" in line:
                    volume = float(line.split()[4])
                    self.volume.append(volume)
                elif "POMASS" in line:
                    parts = line.split("=")
                    if len(parts) == 2:
                        masses = parts[1].split()
                        for i in range(len(masses)):
                            mass = float(masses[i])
                            self.element_mass["type"+str(i)] = mass
                elif "ions per type" in line:
                    parts = line.split("=")
                    counts = parts[1].split()
                    for i in range(len(counts)):
                        count = int(counts[i])
                        self.element_count["type"+str(i)] = count
                elif "free  energy   TOTEN" in line:
                    energy = float(line.split()[4])
                    self.energy.append(energy)
                    
                elif "MDALGO =   2" in line:
                    self.SCF = False
                elif "total pressure" in line:
                    pressure = float(line.split()[3])
                    self.pressure.append(pressure)
                elif ("external pressure" in line) & (self.SCF == True):
                    pressure = float(line.split()[3])
                    self.pressure.append(pressure)
                
                elif 'TOTAL-FORCE' in line:
                    reading_position_force = True
                    next(outcar)
                    continue
                if reading_position_force:
                    if '------' in line:
                        self.position.append(np.array(positions))
                        self.force.append(np.array(forces))
                        positions,forces = [], []
                        reading_position_force = False  
                    else:
                        parts = line.split()
                        position = [float(x) for x in parts[:3]]  # 原子位置
                        positions.append(position)
                        force = [float(x) for x in parts[3:6]]     # 原子受力
                        forces.append(force)

        num_steps = len(self.temperature)
        self.steps = np.arange(1, num_steps + 1) * self.time_step

    def calculate_density(self):
        if len(self.volume) == 0:
            print("No volume data available.")
            return
        
        total_mass = sum([self.element_mass[element] * self.element_count[element] for element in self.element_mass])
        
        # 原子质量单位的转换因子（1 amu = 1.66053906660e-24 g）
        atomic_mass_unit_to_gram = 1.66053906660e-24
        # Calculate density in g/cm³
        density = [(total_mass * atomic_mass_unit_to_gram) / (volume * 1e-24) for volume in self.volume]

        return density