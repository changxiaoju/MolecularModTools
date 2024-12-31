import sys
import numpy as np
from multiprocessing import Pool
from scipy.integrate import cumtrapz
from Analysis.utils import correlationfunction
from Analysis.fit import fit
from scipy.constants import k,eV
from OutputInfo import LammpsMDInfo


class TherCondCalc:

    def runTherCond(
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
        This function calculates average and standard deviation of the thermal conductivity and fit the result with
        single or double-exponential function.

            Parameters:
                -----------
            thermo_df : DataFrame
                thermo dataframe read from log file, to compute thermal conductivity, it should contain heat flux named as c_J[1], c_J[2], c_J[3]

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
        output["Thermal Conductivity"] = {}
        output["Thermal Conductivity"]["Units"] = "W/(mK)"
        if fileprefix == None:
            fileprefix = "./"

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
            if len(thercondo) < trjlen:
                trjlen = len(thercondo)
            thercond[i, :trjlen] = thercondo
            hcacf[i, : trjlen + 1] = autocorrelation
            if ver >= 1:
                sys.stdout.write("\rThermal Conductivity Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        output["Thermal Conductivity"]["Time"] = Time[:trjlen]
        hcacf_mean = np.mean(hcacf, axis=0)
        output["Thermal Conductivity"]["HCACF"] = hcacf
        output["Thermal Conductivity"]["HCACF Average"] = hcacf_mean

        # fit
        fitthercond = fit()
        ave_thercond = np.mean(thercond, axis=0)
        stddev_thercond = np.std(thercond, axis=0)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]
        Value, fitcurve, fitcut = fitthercond.fit(Time, ave_thercond, stddev_thercond, use_double_exp, popt2, std_perc, endt)

        output["Thermal Conductivity"]["Integrals"] = thercond
        output["Thermal Conductivity"]["Average Value"] = Value
        output["Thermal Conductivity"]["Average Integral"] = ave_thercond
        output["Thermal Conductivity"]["Standard Deviation"] = stddev_thercond
        output["Thermal Conductivity"]["Fit"] = fitcurve
        output["Thermal Conductivity"]["Fit Cut"] = fitcut

        return output

    def getthercond(self, thermo_df, Nskip, dt):
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
