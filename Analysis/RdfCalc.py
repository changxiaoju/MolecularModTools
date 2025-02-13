import copy
import numpy as np
from typing import List, Dict, Optional, Union, Tuple


class RdfCalc:
    def runRdf(
        self,
        comx: np.ndarray,
        comy: np.ndarray,
        comz: np.ndarray,
        Lx: float,
        Ly: float,
        Lz: float,
        nummoltype: List[int],
        moltype: List[int],
        namemoltype: List[str],
        stable_steps: int,
        binsize: float,
        maxr: Optional[float] = None,
        output: Optional[Dict] = None,
    ) -> Dict:
        """
        This function calculates the radial distribution function between the
        center of mass for all species in the system

        Parameters:
            comx, comy, comz: Center of mass coordinates
            Lx, Ly, Lz: Box dimensions
            nummoltype: Number of molecules of each type
            moltype: List indicating the type of molecules
            namemoltype: List of molecule labels
            stabel_steps: Number of frames to use after system relaxation
            binsize: Size of bins for RDF calculation
            maxr: Maximum radius for RDF calculation
            output: Optional dictionary to store results

        Returns:
            Dict: Updated output dictionary containing RDF results
        """
        if output is None:
            output = {}
            
        (maxr, numbins, count, g, firststep) = self.setgparam(
            Lx, Ly, Lz, stable_steps, namemoltype, maxr, binsize, len(comx)
        )
        (count) = self.radialdistribution(
            g, len(comx[1]), moltype, comx, comy, comz, Lx, Ly, Lz, binsize, numbins, maxr, count
        )
        (radiuslist) = self.radialnormalization(numbins, binsize, Lx, Ly, Lz, nummoltype, count, g, firststep)
        self.append_dict(radiuslist, g, output, namemoltype)
        return output

    def setgparam(
        self,
        Lx: float,
        Ly: float,
        Lz: float,
        stable_steps: int,
        namemoltype: List[str],
        maxr: Optional[float],
        binsize: float,
        numsteps: int
    ) -> Tuple[float, int, int, np.ndarray, int]:
        # Calculate maximum radius if not specified
        firststep = numsteps - stable_steps
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
        moltype: List[int],
        comx: np.ndarray,
        comy: np.ndarray,
        comz: np.ndarray,
        Lx: float,
        Ly: float,
        Lz: float,
        binsize: float,
        numbins: int,
        maxr: float,
        count: int
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
        output["RDF"] = {}
        output["RDF"]["Units"] = "unitless, angstroms"
        for i in range(0, len(namemoltype)):
            for j in range(i, len(namemoltype)):
                if not all([v == 0 for v in g[i][j]]):
                    output["RDF"]["{0}-{1}".format(namemoltype[i], namemoltype[j])] = copy.deepcopy(g[i][j].tolist())
        if "distance" not in list(output["RDF"].keys()):
            output["RDF"]["Distance"] = copy.deepcopy(radiuslist.tolist())
