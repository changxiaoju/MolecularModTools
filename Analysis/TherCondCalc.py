import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import cumulative_trapezoid
from Analysis.utils import correlationfunction
from Analysis.fit import fit
from scipy.constants import k, eV
from OutputInfo import LammpsMDInfo
from typing import List, Dict, Optional, Union, Tuple
from collections import defaultdict

class TherCondCalc:

    def runTherCond(
        self,
        fileprefix: str,
        Nmd: int,
        Nskip: int,
        use_double_exp: bool,
        logname: str = "log.lammps",
        output: Optional[Dict] = None,
        popt2: Optional[List[float]] = None,
        endt: Optional[float] = None,
        std_perc: Optional[float] = None,
        ver: int = 1,
    ) -> Dict:
        """
        Calculate average and standard deviation of the thermal conductivity and fit
        the result with single or double-exponential function.

        Parameters:
            fileprefix: Path prefix for input files
            Nmd: Number of MD simulations
            Nskip: Initial frames to skip
            use_double_exp: Whether to use double exponential fit
            logname: Name of log file
            output: Optional dictionary to store results
            popt2: Initial guess values for fitting
            endt: Cut time
            std_perc: Standard deviation percentage for cutoff
            ver: Verbosity level

        Returns:
            Dict: Updated output dictionary containing thermal conductivity results
        """
        if output is None:
            output = {}

        logfilename = fileprefix + "000/" + logname
        thermo_df = LammpsMDInfo.thermo_info(logfilename)
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        num_dims = 4  # x,y,z,total
        (Time, thercond_components, autocorrelation) = self.getthercond(thermo_df, Nskip, dt, num_dims)
        trjlen = len(Time)
        
        thercond = np.zeros((num_dims, Nmd, trjlen))
        hcacf = np.zeros((num_dims, Nmd, trjlen + 1))  # x,y,z,total
        
        thercond[-1, 0] = thercond_components[-1]  # total
        hcacf[:, 0, :] = autocorrelation

        if ver >= 1:
            sys.stdout.write("Thermal Conductivity Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            logfilename = fileprefix + str(i).zfill(3) + "/" + logname
            thermo_df = LammpsMDInfo.thermo_info(logfilename)
            (Time, thercondo_components, autocorrelation) = self.getthercond(thermo_df, Nskip, dt, num_dims)
            trjlen = len(Time)
            thercond[:-1, i, :trjlen] = thercondo_components[:3]
            thercond[-1, i, :trjlen] = thercondo_components[-1]
            hcacf[:, i, :] = autocorrelation

            if ver >= 1:
                sys.stdout.write("Thermal Conductivity Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        hcacf_mean = np.mean(hcacf, axis=1)

        fitthercond = fit()
        
        ave_thercond = np.mean(thercond, axis=1)
        stddev_thercond = np.std(thercond, axis=1)
        Value = np.zeros(num_dims)
        fitcurve = np.zeros((num_dims, trjlen))
        fitcut = np.zeros(num_dims)

        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]

        for i in range(num_dims):
            Value[i], fitcurve[i], fitcut[i] = fitthercond.fit(
                Time,
                ave_thercond[i],
                stddev_thercond[i],
                use_double_exp,
                popt2,
                std_perc,
                endt,
            )

        return self.append_dict(
            output, Time, hcacf,hcacf_mean, thercond, ave_thercond, stddev_thercond, Value, fitcurve, fitcut
        )

    def getthercond(
        self,
        thermo_df: pd.DataFrame,
        Nskip: int,
        dt: float,
        num_dims: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a1 = thermo_df["c_J[1]"][Nskip:]
        a2 = thermo_df["c_J[2]"][Nskip:]
        a3 = thermo_df["c_J[3]"][Nskip:]
        
        autocorrelation = np.array([
            correlationfunction(a1, a1),
            correlationfunction(a2, a2),
            correlationfunction(a3, a3),
            np.mean([correlationfunction(a1, a1), correlationfunction(a2, a2), correlationfunction(a3, a3)], axis=0)
        ])
        
        temp = np.mean(thermo_df["Temp"][Nskip:])
        volume = np.mean(thermo_df["Volume"][Nskip:])
        
        thercond_components = np.zeros((num_dims, len(autocorrelation[0])-1))  # x,y,z,total
        for dim in range(num_dims):
            integral = cumulative_trapezoid(autocorrelation[dim], thermo_df['Step'][:len(autocorrelation[dim])]) \
                      * dt / (volume * temp**2 * k / eV) # kB unit: J/K --> eV/K
            thercond_components[dim] = integral * (eV / (1e-12 * 1e-10)) # eV/(ps*A*K) --> J/(s*m*K)
        
        Time = np.array(thermo_df["Step"][: len(autocorrelation[0]) - 1]) * dt
        return (Time, thercond_components, autocorrelation)

    def append_dict(
        self,
        output: Dict,
        Time: np.ndarray,
        hcacf: np.ndarray,
        hcacf_mean: np.ndarray,
        thercond: np.ndarray,
        ave_thercond: np.ndarray,
        stddev_thercond: np.ndarray,
        Value: np.ndarray,
        fitcurve: np.ndarray,
        fitcut: np.ndarray,
    ) -> Dict:
        if "k" not in output:
            output["k"] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        output["k"]["Units"] = "W/mK"
        output["k"]["Time"] = Time.tolist()        
        dim_names = ['x', 'y', 'z', 'total']
        for i, dim in enumerate(dim_names):
            output["k"]["HCACF"][dim] = hcacf[i].tolist()
            output["k"]["HCACF Average"][dim] = hcacf_mean[i].tolist()
            output["k"]["Integrals"][dim] = thercond[i].tolist()
            output["k"]["Average Value"][dim] = float(Value[i])
            output["k"]["Average Integral"][dim] = ave_thercond[i].tolist()
            output["k"]["Standard Deviation"][dim] = stddev_thercond[i].tolist()
            output["k"]["Fit"][dim] = fitcurve[i].tolist()
            output["k"]["Fit Cut"][dim] = int(fitcut[i])
        
        return output
