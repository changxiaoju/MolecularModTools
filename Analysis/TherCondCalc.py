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
        (Time, thercond_components, thercond_total, autocorrelation) = self.getthercond(thermo_df, Nskip, dt)
        trjlen = len(Time)
        thercond = {
            'x': np.zeros((Nmd, trjlen)),
            'y': np.zeros((Nmd, trjlen)),
            'z': np.zeros((Nmd, trjlen)),
            'total': np.zeros((Nmd, trjlen))
        }
        hcacf = {
            'x': np.zeros((Nmd, trjlen + 1)),
            'y': np.zeros((Nmd, trjlen + 1)),
            'z': np.zeros((Nmd, trjlen + 1))
        }
        thercond['total'][:trjlen] = thercond_total
        hcacf['x'][:trjlen] = autocorrelation['x']
        hcacf['y'][:trjlen] = autocorrelation['y']
        hcacf['z'][:trjlen] = autocorrelation['z']
        if ver >= 1:
            sys.stdout.write("Thermal Conductivity Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            logfilename = fileprefix + str(i).zfill(3) + "/" + logname
            thermo_df = LammpsMDInfo.thermo_info(logfilename)
            (Time, thercondo_components, thercondo_total, autocorrelation) = self.getthercond(thermo_df, Nskip, dt)
            trjlen = len(Time)
            for dim in ['x', 'y', 'z']:
                thercond[dim][i, :trjlen] = thercondo_components[dim]
                hcacf[dim][i, :trjlen+1] = autocorrelation[dim]
            thercond['total'][i, :trjlen] = thercondo_total
            if ver >= 1:
                sys.stdout.write("\rThermal Conductivity Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        output["k"]["Time"] = Time[:trjlen]
        output["k"]["HCACF"] = {
            'x': hcacf['x'],
            'y': hcacf['y'],
            'z': hcacf['z'],
            'average': np.mean([hcacf[d] for d in ['x','y','z']], axis=0)
        }
        output["k"]["Integrals"] = {
            'x': thercond['x'],
            'y': thercond['y'],
            'z': thercond['z'],
            'total': thercond['total']
        }

        # fit
        fitthercond = fit()
        for dim in ['x', 'y', 'z', 'total']:
            ave_thercond = np.mean(thercond[dim], axis=0)
            stddev_thercond = np.std(thercond[dim], axis=0)
            if popt2 is None:
                if use_double_exp:
                    popt2 = [2e-3, 5e-2, 2e3, 2e2]
                else:
                    popt2 = [1e-4, 1e2]
            Value, fitcurve, fitcut = fitthercond.fit(
                Time, ave_thercond, stddev_thercond, use_double_exp, popt2, std_perc, endt
            )

            output["k"].setdefault("Average Value", {})[dim] = Value
            output["k"].setdefault("Average Integral", {})[dim] = ave_thercond
            output["k"].setdefault("Standard Deviation", {})[dim] = stddev_thercond
            output["k"].setdefault("Fit", {})[dim] = fitcurve
            output["k"].setdefault("Fit Cut", {})[dim] = fitcut

        return output

    def getthercond(
        self,
        thermo_df: pd.DataFrame,
        Nskip: int,
        dt: float
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray]]:
        numtimesteps = len(thermo_df["Pxy"])
        a1 = thermo_df["c_J[1]"][Nskip:]
        a2 = thermo_df["c_J[2]"][Nskip:]
        a3 = thermo_df["c_J[3]"][Nskip:]
        
        aJ = {
            'x': correlationfunction(a1, a1),
            'y': correlationfunction(a2, a2),
            'z': correlationfunction(a3, a3)
        }
        
        temp = np.mean(thermo_df["Temp"][Nskip:])
        volume = np.mean(thermo_df["Volume"][Nskip:])
        
        thercond_components = {}
        for dim in ['x', 'y', 'z']:
            integral = cumulative_trapezoid(aJ[dim], thermo_df['Step'][:len(aJ[dim])]) * dt / (volume * temp**2 * k / eV)
            thercond_components[dim] = integral * (eV / (1e-12 * 1e-10))
        
        thercond_total = (thercond_components['x'] + thercond_components['y'] + thercond_components['z']) / 3
        
        Time = np.array(thermo_df["Step"][: len(aJ['x']) - 1]) * dt
        return (Time, thercond_components, thercond_total, aJ)
