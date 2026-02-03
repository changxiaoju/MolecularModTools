import numpy as np
from scipy import stats
from tqdm import tqdm
import warnings
from typing import List, Dict, Union, Optional

class DcoeffMsdCalc:

    def runDcoeffMsd(
        self, 
        namemoltype: List[str], 
        dt: float, 
        output: Dict,
        tol: float = 0.075, 
        ver: bool = True
    ) -> Dict:
        """
        Calculate diffusion coefficients from MSD data.
        Always calculates averaged diffusion coefficients by molecule type.
        If output contains 'MSD_i', also calculates per-atom diffusion coefficients.

        Parameters:
            namemoltype: List of molecule labels
            dt: Time step between frames
            output: Dictionary containing MSD results
            tol: Tolerance for finding linear region in log-log plot

        Returns:
            Dict: Updated output dictionary containing diffusion coefficient results
        """
        if ver := True: print("Calculating Averaged Diffusion Coefficients...")
        
        output["D_s_MSD"] = {"units": "m^2/s"}
        time = np.array(output["MSD"]["Time"])
        
        lntime = np.log(time[1:])
        
        for mol_name in namemoltype:
            if mol_name not in output["MSD"]:
                continue
                
            output["D_s_MSD"][mol_name] = {}
            
            for dim_name in ["x", "y", "z", "total"]:
                msd_val = np.array(output["MSD"][mol_name][dim_name])
                
                lnMSD = np.log(msd_val[1:])
                
                firststep = self.findlinearregion(lnMSD, lntime, dt, tol)
                diffusivity = self.getdiffusivity(time, msd_val, firststep, dim_name)
    
                output["D_s_MSD"][mol_name][dim_name] = diffusivity
                output["D_s_MSD"][mol_name][f"{dim_name}_fit_start_ps"] = float(time[firststep])

        if "MSD_i" in output:
            if ver := True: print("Calculating Averaged Diffusion Coefficients...")
            
            msd_data = np.array(output["MSD_i"]["Data"])
            n_atoms, n_steps = msd_data.shape
            
            d_coeffs = np.zeros(n_atoms)
            start_times = np.zeros(n_atoms)
            
            for i in tqdm(range(n_atoms), desc="Fitting Atoms"):
                atom_msd = msd_data[i, :]
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    atom_lnMSD = np.log(atom_msd[1:])
                
                if np.isnan(atom_lnMSD).any():
                    idx_start = int(n_steps / 2)
                else:
                    idx_start = self.findlinearregion(atom_lnMSD, lntime, dt, tol)
                
                fit_time = time[idx_start:]
                fit_msd = atom_msd[idx_start:]
                
                if len(fit_time) < 3:
                    d_val = 0.0
                else:
                    slope = self._fast_slope(fit_time, fit_msd)
                    
                    d_val = (slope * 1e-8) / 6.0 # D = slope / 6, 1 A^2/ps = 1e-8 m^2/s
                
                d_coeffs[i] = d_val
                start_times[i] = time[idx_start]

            output["D_s_MSD_i"] = {
                "units": "m^2/s",
                "values": d_coeffs.tolist(),
                "fit_start_times": start_times.tolist()
            }

        return output

    def findlinearregion(
        self, 
        lnMSD: np.ndarray, 
        lntime: np.ndarray, 
        dt: float, 
        tol: float
    ) -> int:
        """
        Find linear region by scanning backwards from the end of trajectory.
        Searches for the longest region where log-log slope is close to 1.

        Parameters:
            lnMSD: Natural logarithm of MSD values
            lntime: Natural logarithm of time values
            dt: Time step between frames
            tol: Tolerance for slope deviation from 1.0

        Returns:
            int: Starting index of the linear region
        """
        maxtime = len(lnMSD)
        timestepskip = max(1, int(np.ceil(1.0 / dt))) 
        
        numskip = 1
        
        best_start_idx = int(maxtime / 2)
        
        while True:
            t1 = maxtime - 1 - (numskip - 1) * timestepskip
            t2 = maxtime - 1 - numskip * timestepskip
            
            if t2 < 10:
                return 0
            
            denom = lntime[t1] - lntime[t2]
            if denom == 0: 
                numskip += 1
                continue
                
            slope = (lnMSD[t1] - lnMSD[t2]) / denom
            
            if abs(slope - 1.0) < tol:
                best_start_idx = t2
                numskip += 1
            else:
                return best_start_idx

    def getdiffusivity(
        self, 
        Time: np.ndarray, 
        MSD: np.ndarray, 
        firststep: int, 
        dim_name: str
    ) -> float:
        """
        Calculate diffusion coefficient from averaged MSD using linear regression.

        Parameters:
            Time: Time array
            MSD: Mean squared displacement array
            firststep: Starting index for linear regression
            dim_name: Dimension name ("x", "y", "z", or "total")

        Returns:
            float: Diffusion coefficient in m^2/s
        """
        calctime = Time[firststep:]
        calcMSD = MSD[firststep:]
        
        if len(calctime) < 2:
            return 0.0
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(calctime, calcMSD)
        
        conversion = 1e-8
        
        if dim_name == "total":
            diffusivity = (slope * conversion) / 6.0
        else:
            diffusivity = (slope * conversion) / 2.0
            
        return diffusivity

    def _fast_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Fast linear regression to calculate slope.
        Used for per-atom calculations to speed up the inner loop.
        Formula: slope = (N*sum(xy) - sum(x)*sum(y)) / (N*sum(x^2) - sum(x)^2)

        Parameters:
            x: Independent variable array
            y: Dependent variable array

        Returns:
            float: Slope of the linear regression
        """
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / denom
        return slope