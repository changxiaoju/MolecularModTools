import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import cumtrapz
from Analysis.utils import correlationfunction
from Analysis.fit import fit
from scipy.constants import k, eV
from OutputInfo import LammpsMDInfo
from typing import List, Dict, Optional, Union, Tuple


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
            
        output["k"] = {}
        output["k"]["Units"] = "W/(mK)"

        logfilename = fileprefix + "000/" + logname
        thermo_df = LammpsMDInfo.thermo_info(logfilename)
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        # calculate
        (Time, thercondo, autocorrelation) = self.getthercond(thermo_df, Nskip, dt)
        trjlen = len(Time)
        thercond = np.zeros((Nmd, trjlen))
        hcacf = np.zeros((Nmd, trjlen + 1))
        thercond[0] = thercondo
        hcacf[0] = autocorrelation
        if ver >= 1:
            sys.stdout.write("Thermal Conductivity Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            logfilename = fileprefix + str(i).zfill(3) + "/" + logname
            thermo_df = LammpsMDInfo.thermo_info(logfilename)
            (Time, thercondo, autocorrelation) = self.getthercond(thermo_df, Nskip, dt)
            trjlen = len(Time)
            thercond[i, :trjlen] = thercondo
            hcacf[i, : trjlen + 1] = autocorrelation
            if ver >= 1:
                sys.stdout.write("\rThermal Conductivity Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        output["k"]["Time"] = Time[:trjlen]
        hcacf_mean = np.mean(hcacf, axis=0)
        output["k"]["HCACF"] = hcacf
        output["k"]["HCACF Average"] = hcacf_mean

        # fit
        fitthercond = fit()
        ave_thercond = np.mean(thercond, axis=0)
        stddev_thercond = np.std(thercond, axis=0)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]
        Value, fitcurve, fitcut = fitthercond.fit(
            Time, ave_thercond, stddev_thercond, use_double_exp, popt2, std_perc, endt
        )

        output["k"]["Integrals"] = thercond
        output["k"]["Average Value"] = Value
        output["k"]["Average Integral"] = ave_thercond
        output["k"]["Standard Deviation"] = stddev_thercond
        output["k"]["Fit"] = fitcurve
        output["k"]["Fit Cut"] = fitcut

        return output

    def getthercond(
        self,
        thermo_df: pd.DataFrame,
        Nskip: int,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        numtimesteps = len(thermo_df["Pxy"])
        a1 = thermo_df["c_J[1]"][Nskip:]
        a2 = thermo_df["c_J[2]"][Nskip:]
        a3 = thermo_df["c_J[3]"][Nskip:]
        aJ = []
        for a in [a1, a2, a3]:
            aJ.append(correlationfunction(a, a))
        autocorrelation = (aJ[0] + aJ[1] + aJ[2]) / 3
        temp = np.mean(thermo_df["Temp"][Nskip:])
        volume = np.mean(thermo_df["Volume"][Nskip:])

        # Dt = dt * dump_frec
        # Moving forward, we will directly use 'Step' in calculations, inherently including 'dump_frec', thus rendering this step unnecessary
        # fmt: off
        thercondo = (cumtrapz(autocorrelation,
                          thermo_df['Step'][:len(autocorrelation)])) * dt/ (volume * temp * temp * k / eV) # kB unit: J/K --> eV/K
        thercondo = thercondo *(eV / (1e-12 * 1e-10)) # eV/(ps*A*K) --> J/(s*m*K)
        # fmt: on
        Time = np.array(thermo_df["Step"][: len(autocorrelation) - 1]) * dt

        return (Time, thercondo, autocorrelation)
