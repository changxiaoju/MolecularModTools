import numpy as np
import sys, copy
from collections import defaultdict
from scipy.integrate import cumulative_trapezoid
from OutputInfo import ReadBox, LammpsMDInfo
from Analysis.utils import correlationfunction
from Analysis.fit import fit
from typing import List, Dict, Optional, Union, Tuple


class DcoeffVacfCalc:

    def runDcoeffVacf(
        self,
        fileprefix: str,
        namemoltype: List[str],
        Nmd: int,
        Nskip: int,
        interval: int,
        use_double_exp: bool,
        logname: str = "log.lammps",
        velname: str = "dump.vel",
        output: Optional[Dict] = None,
        popt2: Optional[List[float]] = None,
        endt: Optional[float] = None,
        std_perc: Optional[float] = None,
        ver: int = 1,
    ) -> Dict:
        """
        Calculate average and standard deviation of the diffusion coefficient through
        velocity autocorrelation function and fit the result with single or
        double-exponential function.

        Parameters:
            fileprefix: Path prefix for input files
            namemoltype: List of molecule type names
            Nmd: Number of MD simulations
            Nskip: Initial frames to skip
            interval: Reading interval for velocity dump data
            use_double_exp: Whether to use double exponential fit
            logname: Name of log file
            velname: Name of velocity dump file
            output: Optional dictionary to store results
            popt2: Initial guess values for fitting
            endt: Cut time
            std_perc: Standard deviation percentage for cutoff
            ver: Verbosity level

        Returns:
            Dict: Updated output dictionary containing diffusion coefficients
        """
        if output is None:
            output = {}
        # read
        properties = ["vx", "vy", "vz"]

        logfilename = fileprefix + "000/" + logname
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        velfilename = fileprefix + "000/" + velname
        header, frames, atom_type = ReadBox.read_lammps_dump(velfilename, interval)
        steps, dumpinfo, bounds_matrices = zip(*frames)
        velocity = ReadBox.extract_dump_data(dumpinfo, properties)
        vel = velocity.transpose(2, 1, 0)
        nummoltype = np.unique(atom_type, return_counts=True)[1]

        # calculate
        (Time, dcoeffo, autocorrelation) = self.getdcoeff(vel, Nskip, dt, nummoltype, dump_frec, interval)

        trjlen = len(Time)
        Ndim = vel.shape[0]
        Nmoltype = len(nummoltype)

        dcoeff = np.zeros((Nmoltype, Ndim + 1, Nmd, trjlen))
        vacf = np.zeros((Nmoltype, Ndim + 1, Nmd, trjlen + 1))
        dcoeff[:, :, 0, :] = dcoeffo
        vacf[:, :, 0, :] = autocorrelation

        if ver >= 1:
            sys.stdout.write("Dcoeff Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            velfilename = fileprefix + str(i).zfill(3) + "/" + velname
            header, frames, atom_type = ReadBox.read_lammps_dump(velfilename, interval)
            steps, dumpinfo, bounds_matrices = zip(*frames)
            velocity = ReadBox.extract_dump_data(dumpinfo, properties)
            vel = velocity.transpose(2, 1, 0)

            (Time, dcoeffo, autocorrelation) = self.getdcoeff(vel, Nskip, dt, nummoltype, dump_frec, interval)
            trjlen = len(Time)
            dcoeff[:, :, i, :trjlen] = dcoeffo
            vacf[:, :, i, : trjlen + 1] = autocorrelation

            if ver >= 1:
                sys.stdout.write("Dcoeff Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        vacf_mean = np.mean(vacf, axis=2)

        # fit
        fitdcoeff = fit()
        ave_dcoeff = np.mean(dcoeff, axis=2)
        stddev_dcoeff = np.std(dcoeff, axis=2)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]

        Value = np.zeros((ave_dcoeff.shape[0], ave_dcoeff.shape[1]))
        fitcurve = np.zeros_like(ave_dcoeff)
        fitcut = np.zeros_like(Value)
        for i in range(ave_dcoeff.shape[0]):
            for j in range(ave_dcoeff.shape[1]):
                Value[i, j], fitcurve[i, j, :], fitcut[i, j] = fitdcoeff.fit(
                    Time,
                    ave_dcoeff[i, j],
                    stddev_dcoeff[i, j],
                    use_double_exp,
                    popt2,
                    std_perc,
                    endt,
                )

        output = self.append_dict(
            Time,
            namemoltype,
            output,
            vacf,
            vacf_mean,
            dcoeff,
            ave_dcoeff,
            stddev_dcoeff,
            Value,
            fitcurve,
            fitcut,
        )
        return output

    def getdcoeff(
        self, vel: np.ndarray, Nskip: int, dt: float, nummoltype: np.ndarray, dump_frec: int, interval: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        vel = vel[:, :, Nskip:]
        Ndim = vel.shape[0]
        Ndumps = vel.shape[2]
        autocorrelation = np.zeros((len(nummoltype), Ndim + 1, Ndumps))

        # Calculate the autocorrelation of each species in each dimension
        for d in range(Ndim):
            atomindex_start = 0
            atomindex_end = 0
            for typeindex, num in enumerate(nummoltype):
                atomindex_end += num
                tmp_autocorrelation = np.zeros(Ndumps)
                for atomindex in range(atomindex_start, atomindex_end):
                    atomindex_autocorrelation = correlationfunction(vel[d, atomindex, :], vel[d, atomindex, :])

                    tmp_autocorrelation += atomindex_autocorrelation

                autocorrelation[typeindex, d, :] = tmp_autocorrelation / num
                autocorrelation[typeindex, -1, :] += tmp_autocorrelation / num
                atomindex_start += num

        Dt = dt * dump_frec * interval
        diffuso = (
            cumulative_trapezoid(autocorrelation, dx=Dt) / 300000000
        )  # m^2/s,  (A^2/ps)/3 = 10^(-20)/10^(-12)/3 = 1/(3*10^8)
        Time = np.arange(diffuso.shape[2]) * Dt

        return (Time, diffuso, autocorrelation)

    def append_dict(
        self,
        Time: np.ndarray,
        namemoltype: List[str],
        output: Dict,
        vacf: np.ndarray,
        vacf_mean: np.ndarray,
        dcoeff: np.ndarray,
        ave_dcoeff: np.ndarray,
        stddev_dcoeff: np.ndarray,
        Value: np.ndarray,
        fitcurve: np.ndarray,
        fitcut: np.ndarray,
    ) -> Dict:
        if "D_s_VACF" not in output:
            output["D_s_VACF"] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        output["D_s_VACF"]["Units"] = "m^2/s"
        output["D_s_VACF"]["Time"] = Time
        dim = ["x", "y", "z", "total"]
        for i in range(ave_dcoeff.shape[0]):
            for j in range(ave_dcoeff.shape[1]):
                output["D_s_VACF"]["VACF"][namemoltype[i]][dim[j]] = copy.deepcopy(vacf[i, j])
                output["D_s_VACF"]["VACF Average"][namemoltype[i]][dim[j]] = copy.deepcopy(vacf_mean[i, j])
                output["D_s_VACF"]["Integrals"][namemoltype[i]][dim[j]] = copy.deepcopy(dcoeff[i, j])
                output["D_s_VACF"]["Average Value"][namemoltype[i]][dim[j]] = copy.deepcopy(Value[i, j])
                output["D_s_VACF"]["Average Integral"][namemoltype[i]][dim[j]] = copy.deepcopy(ave_dcoeff[i, j])
                output["D_s_VACF"]["Standard Deviation"][namemoltype[i]][dim[j]] = copy.deepcopy(stddev_dcoeff[i, j])
                output["D_s_VACF"]["Fit"][namemoltype[i]][dim[j]] = copy.deepcopy(fitcurve[i, j])
                output["D_s_VACF"]["Fit Cut"][namemoltype[i]][dim[j]] = copy.deepcopy(fitcut[i, j])

        return output
