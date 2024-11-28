import sys
import numpy as np
from multiprocessing import Pool
from scipy.integrate import cumtrapz
from Analysis.utils import correlationfunction
from Analysis.fit import fit

from OutputInfo import LammpsMDInfo


class ViscCalc:

    def runVisc(
        self,
        fileprefix,
        Nmd,
        Nskip,
        use_double_exp,
        logname="log.lammps",
        output={},
        popt2=None,
        endt=None,
        std_perc=None,
        ver=1,
    ):
        """
        This function calculates average and standard deviation of the viscosity and fit the result with
        single or double-exponential function.

            Parameters:
                -----------
            thermo_df : DataFrame
                thermo dataframe read from log file, to compute viscosity, it should contain pressure tensor

            fileprefix : str

            Nmd : int
                num of md

            Nskip: int
                initial lines ignored during the calculation

            use_double_exp : bool
                weather use double-exponential fit

            logname : str, optional

            output : dict, optional

            popt2 : list of float, optional
                initial guess value, if None, use [1e-4,1e2] for single-exponential fit, [1e-3,1.5e-1,1e2,1e3] for double-exponential fit

            endt ：float, optional
                cut time

            std_perc : float, optional
                "It was found empirically that the time at which the calculated standard deviation σ(t) was about 40% of the corresponding viscosity (rough average of the flat region in the running integral) was a good choice for tcut."
                https://pubs.acs.org/doi/10.1021/acs.jctc.5b00351.

                if endt=None, then use std_prec, if std_prec=None, then std_prec=0.4

            ver: int, optional
                if ver>1, output the progress
        """
        # read
        output["Viscosity"] = {}
        output["Viscosity"]["Units"] = "mcP"
        if fileprefix == None:
            fileprefix = "./"

        logfilename = fileprefix + "000/" + logname
        thermo_df = LammpsMDInfo.thermo_info(logfilename)
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        # calculate
        (Time, visco, autocorrelation) = self.getvisc(thermo_df, Nskip, dt, NCORES)
        trjlen = len(Time)
        viscosity = np.zeros((Nmd, trjlen))
        sacf = np.zeros((Nmd, trjlen + 1))
        viscosity[0] = visco
        sacf[0] = autocorrelation
        if ver >= 1:
            sys.stdout.write("Viscosity Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            logfilename = fileprefix + str(i).zfill(3) + "/" + logname
            thermo_df = LammpsMDInfo.thermo_info(logfilename)
            (Time, visco, autocorrelation) = self.getvisc(thermo_df, Nskip, dt, NCORES)
            if len(visco) < trjlen:
                trjlen = len(visco)
            viscosity[i, :trjlen] = visco
            sacf[i, : trjlen + 1] = autocorrelation
            if ver >= 1:
                sys.stdout.write("\rViscosity Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        output["Viscosity"]["Time"] = Time[:trjlen]
        sacf_mean = np.mean(sacf, axis=0)
        output["Viscosity"]["sacf"] = sacf
        output["Viscosity"]["sacf Average"] = sacf_mean

        # fit
        fitvisc = fit()
        ave_visc = np.mean(viscosity, axis=0)
        stddev_visc = np.std(viscosity, axis=0)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]
        Value, fitcurve = fitvisc.fit(Time, ave_visc, stddev_visc, use_double_exp, popt2, std_perc, endt)

        output["Viscosity"]["Integrals"] = viscosity
        output["Viscosity"]["Average Value"] = Value
        output["Viscosity"]["Average Integral"] = ave_visc
        output["Viscosity"]["Standard Deviation"] = stddev_visc
        output["Viscosity"]["Fit"] = fitcurve

        return output

    def getvisc(self, thermo_df, Nskip, dt, NCORES):
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

        # fmt: off
        visco = (cumtrapz(autocorrelation,
                          thermo_df['Step'][:len(autocorrelation)]))*dt*10**-12*1000*101325.**2*thermo_df['Volume'].iloc[-1]*10**-30/(1.38*10**-23*temp)
        # fmt: on
        Time = np.array(thermo_df["Step"][: len(autocorrelation) - 1]) * dt

        return (Time, visco, autocorrelation)
