import numpy as np
import copy
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple, Any


class MsdCalc:

    def runMsd(
        self,
        coordinates: np.ndarray,
        bounds_matrix: np.ndarray,
        moltype: Union[List[int], np.ndarray],
        namemoltype: List[str],
        dt: float,
        skip: int,
        num_init: Optional[int] = None,
        output: Optional[Dict] = None,
        ver: bool = True,
    ) -> Dict:
        """
        This function calculates the mean square displacement for all molecule
        types in the system from center of mass positions

        Parameters:
            coordinates: A Three-dimensional array of floats with shape (Nframes, Natoms, 3)
            bounds_matrix: A two-dimensional array of floats with shape (3, 3)
            moltype: List or numpy array indicating the type of molecules
            namemoltype: List of molecule labels
            dt: Timestep
            skip: Initial frames to skip
            num_init: Number of initial timesteps for MSD calculation
            output: Optional dictionary to store results
            ver: Whether to print progress


        Returns:
            Dict: Updated output dictionary containing MSD results
        """
        if output is None:
            output = {}

        comx, comy, comz = coordinates.transpose(2, 0, 1)
        Lx, Ly, Lz = bounds_matrix[0, 0], bounds_matrix[1, 1], bounds_matrix[2, 2]
        moltype = moltype - np.array(moltype).min() #start from 0! 

        
        Lx2, Ly2, Lz2 = Lx / 2, Ly / 2, Lz / 2
        (comx, comy, comz) = self.unwrap(comx, comy, comz, Lx, Ly, Lz, Lx2, Ly2, Lz2)
        num_timesteps = len(coordinates)
        (num_init, len_MSD, MSD, diffusivity) = self.gettimesteps(num_timesteps, namemoltype, skip, num_init)
        (molcheck, nummol) = self.setmolarray(moltype, namemoltype)
        with tqdm(total=num_init, desc="Calculating MSD", 
                disable=not ver) as pbar:
            for i in range(skip, num_init + skip):
                for j in range(i, i + len_MSD):
                    r2_x = (comx[j] - comx[i]) ** 2
                    r2_y = (comy[j] - comy[i]) ** 2
                    r2_z = (comz[j] - comz[i]) ** 2
                    r2_total = r2_x + r2_y + r2_z
                    
                    MSD = self.MSDadd(r2_x, MSD, molcheck, i, j, 0)  # x dimension
                    MSD = self.MSDadd(r2_y, MSD, molcheck, i, j, 1)  # y dimension
                    MSD = self.MSDadd(r2_z, MSD, molcheck, i, j, 2)  # z dimension
                    MSD = self.MSDadd(r2_total, MSD, molcheck, i, j, 3)  # total
                pbar.update(1)
        MSD = self.MSDnorm(MSD, num_init, nummol)
        Time = self.createtime(dt, len_MSD)
        self.append_dict(MSD, namemoltype, output, Time)
        return output

    def unwrap(
        self,
        comx: np.ndarray,
        comy: np.ndarray,
        comz: np.ndarray,
        Lx: float,
        Ly: float,
        Lz: float,
        Lx2: float,
        Ly2: float,
        Lz2: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # unwraps the coordintes of the molecules
        # assumes if a molecule is more than half a box length away from its
        # previous coordinte that it passed through a periodic boundary
        for i in range(1, len(comx)):
            for j in range(0, len(comx[i])):
                if (comx[i][j] - comx[i - 1][j]) > Lx2:
                    while (comx[i][j] - comx[i - 1][j]) > Lx2:
                        comx[i][j] -= Lx
                elif (comx[i][j] - comx[i - 1][j]) < (-Lx2):
                    while (comx[i][j] - comx[i - 1][j]) < (-Lx2):
                        comx[i][j] += Lx

                if (comy[i][j] - comy[i - 1][j]) > Ly2:
                    while (comy[i][j] - comy[i - 1][j]) > Ly2:
                        comy[i][j] -= Ly
                elif (comy[i][j] - comy[i - 1][j]) < (-Ly2):
                    while (comy[i][j] - comy[i - 1][j]) < (-Ly2):
                        comy[i][j] += Ly

                if (comz[i][j] - comz[i - 1][j]) > Lz2:
                    while (comz[i][j] - comz[i - 1][j]) > Lz2:
                        comz[i][j] -= Lz
                elif (comz[i][j] - comz[i - 1][j]) < (-Lz2):
                    while (comz[i][j] - comz[i - 1][j]) < (-Lz2):
                        comz[i][j] += Lz
        return (comx, comy, comz)

    def gettimesteps(
        self,
        num_timesteps: int,
        namemoltype: List[str],
        skip: int,
        num_init: Optional[int] = None
    ) -> Tuple[int, int, np.ndarray, List]:
        # Calculates the length of the trajectory
        # Uses length to determine length of MSD and number of initial timesteps
        if num_init == None:
            num_init = int(np.floor((num_timesteps - skip) / 2))
        else:
            num_init = int(num_init)

        len_MSD = num_timesteps - skip - num_init
        MSD = np.zeros((len(namemoltype), 4, len_MSD))  # 4 dimensions: x,y,z,total
        diffusivity = []
        return (num_init, len_MSD, MSD, diffusivity)

    def setmolarray(
        self,
        moltype: Union[List[int], np.ndarray],
        namemoltype: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Generates arrays for dot product calculation
        # Array is MxN where M is number of molecule types and N is number of molecules
        # value is 1 if molecule N is of type M else is 0
        molcheck = np.zeros((len(namemoltype), len(moltype)))
        for i in range(0, len(moltype)):
            molcheck[moltype[i]][i] = 1
        nummol = np.zeros(len(namemoltype))
        for i in range(0, len(nummol)):
            nummol[i] = np.sum(molcheck[i])
        return (molcheck, nummol)

        """
        e.g.,
        moltype = [0,1,0,0]
        namemoltype = ['H','He']
        
        molcheck:
        array([[1., 0., 1., 1.],
               [0., 1., 0., 0.]])
        """


    def MSDadd(
        self,
        r2: np.ndarray,
        MSD: np.ndarray,
        molcheck: np.ndarray,
        i: int,
        j: int,
        dim_idx: int  # 0=x,1=y,2=z,3=total
    ) -> np.ndarray:
        # Add dimensional index to MSD accumulation
        for k in range(0, len(molcheck)):
            sr2 = np.dot(r2, molcheck[k])
            MSD[k][dim_idx][j - i] += sr2
        return MSD

    def MSDnorm(
        self,
        MSD: np.ndarray,
        MSDt: int,
        nummol: np.ndarray
    ) -> np.ndarray:
        # Normalize all dimensions
        for i in range(0, len(nummol)):
            for dim in range(4):  # x,y,z,total
                MSD[i][dim] /= MSDt * nummol[i]
        return MSD

    def createtime(
        self,
        dt: float,
        MSDt: int
    ) -> np.ndarray:
        # Creates an array of time values
        Time = np.arange(0, MSDt, dtype=float)
        Time *= dt
        return Time

    def append_dict(
        self,
        MSD: np.ndarray,
        namemoltype: List[str],
        output: Dict,
        Time: np.ndarray
    ) -> None:
        # Modify output structure to include dimensions
        output["MSD"] = {
            "Units": "Angstroms^2, ps",
            "Dimensions": ["x", "y", "z", "total"],
            "Time": copy.deepcopy(Time.tolist())
        }
        
        for i in range(len(namemoltype)):
            output["MSD"][namemoltype[i]] = {
                "x": copy.deepcopy(MSD[i,0].tolist()),
                "y": copy.deepcopy(MSD[i,1].tolist()),
                "z": copy.deepcopy(MSD[i,2].tolist()),
                "total": copy.deepcopy(MSD[i,3].tolist())
            }
