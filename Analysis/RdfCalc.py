import copy
import numpy as np
from tqdm import tqdm
from Analysis.AdjustCell import adjust_cell
from typing import List, Dict, Optional, Union, Tuple


class RdfCalc:
    def runRdf(
        self,
        coordinates: np.ndarray,
        bounds_matrix: np.ndarray,
        moltype: Union[List[int], np.ndarray],
        namemoltype: List[str],
        Nskip: int = 0,
        binsize: float = 0.01,
        maxr: Optional[float] = None,
        output: Optional[Dict] = None,
        replicate: int = 1,
        ver: bool = True,
    ) -> Dict:
        """
        This function calculates the radial distribution function between the
        center of mass for all species in the system

        Parameters:
            coordinates: A Three-dimensional array of floats with shape (Nframes, Natoms, 3)
            bounds_matrix: A two-dimensional array of floats with shape (3, 3)
            moltype: List or numpy array indicating the type of molecules
            namemoltype: List of molecule labels
            Nskip: Number of initial frames to skip (default: 0, uses all frames)
            binsize: Size of bins for RDF calculation
            maxr: Maximum radius for RDF calculation
            output: Optional dictionary to store results
            replicate: Number of times to replicate the system, default is 1
            ver: Boolean to enable/disable progress bar

        Returns:
            Dict: Updated output dictionary containing RDF results
        """
        if output is None:
            output = {}
        
        if replicate > 1:
            new_coordinates = np.zeros((len(coordinates), len(coordinates[0])*replicate*replicate*replicate, 3))
            for i in range(len(coordinates)):
                adj_cell = adjust_cell(coordinates[i], bounds_matrix, moltype)
                new_bounds_matrix, i_coordinates, new_moltype = adj_cell.make_supercell([replicate, replicate, replicate])
                new_coordinates[i] = i_coordinates
            coordinates = np.array(new_coordinates)
            bounds_matrix = new_bounds_matrix
            moltype = new_moltype

        if coordinates.ndim == 2:
            coordinates = coordinates[np.newaxis, :, :]
        
        comx, comy, comz = coordinates.transpose(2, 0, 1)
        Lx, Ly, Lz = bounds_matrix[0, 0], bounds_matrix[1, 1], bounds_matrix[2, 2]
        moltype = moltype - np.array(moltype).min() #start from 0! 


        (maxr, numbins, count, g, firststep) = self.setgparam(
            Lx, Ly, Lz, Nskip, namemoltype, maxr, binsize, len(comx)
        )
        (count) = self.radialdistribution(
            g, len(comx[0]), moltype, comx, comy, comz, Lx, Ly, Lz, binsize, numbins, maxr, count, ver
        )
        nummoltype = np.unique(moltype, return_counts=True)[1]

        (radiuslist) = self.radialnormalization(numbins, binsize, Lx, Ly, Lz, nummoltype, count, g, firststep)
        self.append_dict(radiuslist, g, output, namemoltype)
        return output

    def setgparam(
        self,
        Lx: float,
        Ly: float,
        Lz: float,
        Nskip: int,
        namemoltype: List[str],
        maxr: Optional[float],
        binsize: float,
        numsteps: int
    ) -> Tuple[float, int, int, np.ndarray, int]:
        # Calculate maximum radius if not specified
        firststep = Nskip
        if maxr == None:
            maxr = min(Lx / 2, Ly / 2, Lz / 2)
        else:
            maxr = float(maxr)
        numbins = int(np.ceil(maxr / binsize))
        count = firststep
        g = np.zeros((len(namemoltype), len(namemoltype), numbins))
        return maxr, numbins, count, g, firststep

    def radialdistribution(
        self,
        g: np.ndarray,
        nummol: int,
        moltype: Union[List[int], np.ndarray],
        comx: np.ndarray,
        comy: np.ndarray,
        comz: np.ndarray,
        Lx: float,
        Ly: float,
        Lz: float,
        binsize: float,
        numbins: int,
        maxr: float,
        count: int,
        ver: bool = True
    ) -> int:
        # calculates the number of molecules within each shell
        comxt = np.array(comx[count:]).transpose()
        comyt = np.array(comy[count:]).transpose()
        comzt = np.array(comz[count:]).transpose()
        indexlist = []
        # change indeces order to com*[molecule][timestep]

        for i in range(0, len(g)):
            indexlist.append(np.array(moltype) == i)
            # creates two dimensional array
            # first dimension is molecule type
            # second dimension is over molecules
            # contains boolean for if that molecule is of the molecule type

        with tqdm(total=nummol-1, desc="Calculating RDF", disable=not ver) as pbar:
            for molecule in range(0, nummol - 1):
                dx = comxt[molecule + 1 :] - np.tile(comxt[molecule], (len(comxt) - molecule - 1, 1))
                dy = comyt[molecule + 1 :] - np.tile(comyt[molecule], (len(comyt) - molecule - 1, 1))
                dz = comzt[molecule + 1 :] - np.tile(comzt[molecule], (len(comzt) - molecule - 1, 1))

                dx -= Lx * np.around(dx / Lx)
                dy -= Ly * np.around(dy / Ly)
                dz -= Lz * np.around(dz / Lz)
                # minimum image convention

                r2 = dx**2 + dy**2 + dz**2
                r = np.sqrt(r2)
                for i in range(0, len(indexlist)):
                    gt, dist = np.histogram(
                        r[indexlist[i][molecule + 1 :]].ravel(),
                        bins=numbins,
                        range=(0.5 * binsize, binsize * (numbins + 0.5)),
                    )
                    g[moltype[molecule]][i] += gt
                    g[i][moltype[molecule]] += gt
                pbar.update(1)

        count = len(comx)
        return count

    def radialnormalization(
        self,
        numbins: int,
        binsize: float,
        Lx: float,
        Ly: float,
        Lz: float,
        nummoltype: List[int],
        count: int,
        g: np.ndarray,
        firststep: int
    ) -> np.ndarray:
        # normalizes g to box density
        radiuslist = (np.arange(numbins) + 1) * binsize
        radiuslist = np.around(radiuslist, decimals=3)
        for i in range(0, len(g)):
            for j in range(0, len(g)):
                # fmt: off
                g[i][j] *= Lx * Ly * Lz / nummoltype[i] / nummoltype[j] / 4 / np.pi / ( 
                               radiuslist) ** 2 / binsize / (
                               count - firststep)
                # fmt: on
        return radiuslist

    def append_dict(
        self,
        radiuslist: np.ndarray,
        g: np.ndarray,
        output: Dict,
        namemoltype: List[str]
    ) -> None:
        output["g(r)"] = {}
        output["g(r)"]["Units"] = "unitless, Å"
        for i in range(0, len(namemoltype)):
            for j in range(i, len(namemoltype)):
                if not all([v == 0 for v in g[i][j]]):
                    output["g(r)"]["{0}-{1}".format(namemoltype[i], namemoltype[j])] = g[i][j].tolist()
        if "Distance" not in list(output["g(r)"].keys()):
            output["g(r)"]["Distance"] = radiuslist.tolist()
