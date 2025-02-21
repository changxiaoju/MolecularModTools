import numpy as np
from scipy import stats
import warnings
from typing import List, Dict, Union, Optional


class DcoeffMsdCalc:

    def runDcoeffMsd(
        self, 
        namemoltype: List[str], 
        dt: float, 
        tol: float, 
        output: Dict
    ) -> Dict:
        """
        This function fits the mean square displacement to calculate the
        diffusivity for all molecule types in the system

        Parameters:
            namemoltype: List of molecule type names
            dt: Time step
            tol: Tolerance for linear fitting
            output: Dictionary to store MSD data and also used to store the calculated diffusion coefficients. 
                    It should have appropriate keys for MSD data corresponding to each molecule type.

        Returns:
            Dict: Updated output dictionary containing diffusion coefficients
        """

        output["D_s_MSD"] = {
            "units": "m^2/s",
            "dimensions": ["x", "y", "z", "total"]
        }
        
        dimensions = output["MSD"]["Dimensions"]
        
        for i in range(len(namemoltype)):
            mol_name = namemoltype[i]
            output["D_s_MSD"][mol_name] = {}
            
            for dim_idx, dim_name in enumerate(dimensions):
                MSD_data = output["MSD"][mol_name][dim_name]
                time = output["MSD"]["Time"]
                
                # Skip first point (log(0) undefined)
                lnMSD = np.log(MSD_data[1:])
                lntime = np.log(time[1:])
                
                firststep = self.findlinearregion(lnMSD, lntime, dt, tol)
                diffusivity = self.getdiffusivity(time, MSD_data, firststep)
                
                output["D_s_MSD"][mol_name][dim_name] = diffusivity
        
        return output

    def findlinearregion(
        self, 
        lnMSD: np.ndarray, 
        lntime: np.ndarray, 
        dt: float, 
        tol: float
    ) -> int:
        # Uses the slope of the log-log plot to find linear regoin of MSD
        timestepskip = np.ceil(1 / dt)
        linearregion = True
        maxtime = len(lnMSD)
        numskip = 1
        while linearregion == True:
            if numskip * timestepskip + 1 > maxtime:
                firststep = maxtime - 1 - (numskip - 1) * timestepskip
                return firststep
                linearregion = False
            else:
                t1 = int(maxtime - 1 - (numskip - 1) * timestepskip)
                t2 = int(maxtime - 1 - numskip * timestepskip)
                slope = (lnMSD[t1] - lnMSD[t2]) / (lntime[t1] - lntime[t2])
                if abs(slope - 1.0) < tol:
                    numskip += 1
                else:
                    firststep = t1
                    return firststep
                    linearregion = False

    def getdiffusivity(
        self, 
        Time: np.ndarray, 
        MSD: np.ndarray, 
        firststep: int
    ) -> Union[float, str]:
        # Fits the linear region of the MSD to obtain the diffusivity
        calctime = []
        calcMSD = []
        for i in range(int(firststep), len(Time)):
            calctime.append(Time[i])
            calcMSD.append(MSD[i])
        if len(calctime) == 1:
            diffusivity = "runtime not long enough"
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                line = stats.linregress(calctime, calcMSD)
            slope = line[0]
            diffusivity = slope / 600000000  # m^2/s,  (A^2/ps)/6 = 10^(-20)/10^(-12)/6 = 1/(6*10^8)
        return diffusivity
