import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.integrate import cumulative_trapezoid
from Analysis.utils import correlationfunction
from Analysis.fit import fit
from scipy.constants import k
from OutputInfo import LammpsMDInfo
from typing import List, Dict, Optional, Union, Tuple


class ViscCalc:

    def runVisc(
        self,
        fileprefix: str,
        Nmd: int,
        Nskip: int = 0,
        use_double_exp: bool = True,
        logname: str = "log.lammps",
        output: Optional[Dict] = None,
        popt2: Optional[List[float]] = None,
        endt: Optional[float] = None,
        std_perc: Optional[float] = None,
        ver: int = 1,
    ) -> Dict:
        """
        This function calculates average and standard deviation of the viscosity and fits
        the result with single or double-exponential function.

        Parameters:
            fileprefix: Path prefix for input files
            Nmd: Number of MD simulations
            Nskip: Initial frames to skip (default: 0, uses all frames)
            use_double_exp: Whether to use double-exponential fit
            logname: Name of log file
            output: Optional dictionary to store results
            popt2: Initial guess values for fitting
            endt: Cut time
            std_perc: Standard deviation percentage for cutoff
            ver: Verbosity level

        Returns:
            Dict: Updated output dictionary containing viscosity results
        """
        if output is None:
            output = {}

        output["μ"] = {}
        output["μ"]["Units"] = "mPa·s"

        logfilename = fileprefix + "000/" + logname
        thermo_df = LammpsMDInfo.thermo_info(logfilename)
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        # calculate
        (Time, visco, autocorrelation) = self.getvisc(thermo_df, Nskip, dt)
        trjlen = len(Time)
        viscosity = np.zeros((Nmd, trjlen))
        sacf = np.zeros((Nmd, trjlen + 1))
        viscosity[0] = visco
        sacf[0] = autocorrelation
        if ver >= 1:
            sys.stdout.write("Viscosity 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            logfilename = fileprefix + str(i).zfill(3) + "/" + logname
            thermo_df = LammpsMDInfo.thermo_info(logfilename)
            (Time, visco, autocorrelation) = self.getvisc(thermo_df, Nskip, dt)
            trjlen = len(Time)
            viscosity[i, :trjlen] = visco
            sacf[i, : trjlen + 1] = autocorrelation
            if ver >= 1:
                sys.stdout.write("\rViscosity {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        output["μ"]["Time"] = Time[:trjlen].tolist()
        sacf_mean = np.mean(sacf, axis=0)
        output["μ"]["SACF"] = sacf.tolist()
        output["μ"]["SACF Average"] = sacf_mean.tolist()

        # fit
        fitvisc = fit()
        ave_visc = np.mean(viscosity, axis=0)
        stddev_visc = np.std(viscosity, axis=0)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]
        Value, fitcurve, fitcut = fitvisc.fit(Time, ave_visc, stddev_visc, use_double_exp, popt2, std_perc, endt)

        output["μ"]["Integrals"] = viscosity.tolist()
        output["μ"]["Average Value"] = float(Value)
        output["μ"]["Average Integral"] = ave_visc.tolist()
        output["μ"]["Standard Deviation"] = stddev_visc.tolist()
        output["μ"]["Fit"] = fitcurve.tolist()
        output["μ"]["Fit Cut"] = int(fitcut)

        return output

    def getvisc(self, thermo_df: pd.DataFrame, Nskip: int, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        numtimesteps = len(thermo_df["Pxy"])
        a1 = thermo_df["Pxy"][Nskip:]
        a2 = thermo_df["Pxz"][Nskip:]
        a3 = thermo_df["Pyz"][Nskip:]
        a4 = thermo_df["Pxx"][Nskip:] - thermo_df["Pyy"][Nskip:]
        a5 = thermo_df["Pyy"][Nskip:] - thermo_df["Pzz"][Nskip:]
        a6 = thermo_df["Pxx"][Nskip:] - thermo_df["Pzz"][Nskip:]
        pv = []
        for a in [a1, a2, a3, a4, a5, a6]:
            pv.append(correlationfunction(a, a))
        autocorrelation = (pv[0] + pv[1] + pv[2]) / 6 + (pv[3] + pv[4] + pv[5]) / 12

        temp = np.mean(thermo_df["Temp"][Nskip:])

        # Dt = dt * dump_frec
        # Moving forward, we will directly use 'Step' in calculations, inherently including 'dump_frec', thus rendering this step unnecessary
        # fmt: off
        visco = (cumulative_trapezoid(autocorrelation,
                          x=thermo_df['Step'][:len(autocorrelation)]))*(1e5)**2*dt*(1e-12)*thermo_df['Volume'].iloc[-1]*(1e-30)/(k*temp)*(1e3)
        # 1e5: bar to pascal; 1e-12: ps to s; 1e-30: Angstrom**3 to m**3; k: Boltzmann constant; 1e3: Pa*s to mPa*s
        # fmt: on
        Time = np.array(thermo_df["Step"][: len(autocorrelation) - 1]) * dt

        return (Time, visco, autocorrelation)
