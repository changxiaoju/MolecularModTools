import numpy as np
import sys, copy
from collections import defaultdict
from scipy.integrate import cumulative_trapezoid
from OutputInfo import ReadBox, LammpsMDInfo
from Analysis.utils import correlationfunction
from Analysis.fit import fit
from typing import List, Dict, Optional, Union, Tuple, Any


class MutualDcoeffJrcfCalc:

    def runMutualDcoeffjrcf(
        self,
        fileprefix: str,
        molmass: List[int],
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
        This function calculates average and standard deviation of the mutual diffusion coefficient through
        relative particle current jr correlation function and fit the result with
        single or double-exponential function.

        Parameters:
            fileprefix: Path prefix for input files
            molmass: List of molecular masses
            Nmd: Number of MD simulations
            Nskip: Initial frames to skip
            interval: Reading interval for velocity dump data
            use_double_exp: Whether to use double-exponential fit
            logname: Name of log file
            velname: Name of velocity dump file
            output: Optional dictionary to store results
            popt2: Initial guess values for fitting
            endt: Cut time
            std_perc: Standard deviation percentage for cutoff
            ver: Verbosity level

        Returns:
            Dict: Updated output dictionary containing mutual diffusion coefficients
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
        velosity = ReadBox.extract_dump_data(dumpinfo, properties)
        vel = velosity.transpose(2, 1, 0)
        nummoltype = np.unique(atom_type, return_counts=True)[1]

        # calculate
        (Time, mutual_dcoeffo, correlation) = self.getmuldcoeff(
            vel, Nskip, dt, nummoltype, dump_frec, interval, molmass
        )

        trjlen = len(Time)
        Ndim = vel.shape[0]
        Nmoltype = len(nummoltype)
        Ncorrelation = int((Nmoltype - 1) * (Nmoltype - 1))

        mutual_dcoeff = np.zeros((Ncorrelation, Ndim + 1, Nmd, trjlen))
        jrcf = np.zeros((Ncorrelation, Ndim + 1, Nmd, trjlen + 1))
        mutual_dcoeff[:, :, 0, :] = mutual_dcoeffo
        jrcf[:, :, 0, :] = correlation

        if ver >= 1:
            sys.stdout.write("Mutual Diffusion Coefficient Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            velfilename = fileprefix + str(i).zfill(3) + "/" + velname
            header, frames, atom_type = ReadBox.read_lammps_dump(velfilename, interval)
            steps, dumpinfo, bounds_matrices = zip(*frames)
            velosity = ReadBox.extract_dump_data(dumpinfo, properties)
            vel = velosity.transpose(2, 1, 0)

            (Time, mutual_dcoeffo, correlation) = self.getmuldcoeff(
                vel, Nskip, dt, nummoltype, dump_frec, interval, molmass
            )

            trjlen = len(Time)
            mutual_dcoeff[:, :, i, :trjlen] = mutual_dcoeffo
            jrcf[:, :, i, : trjlen + 1] = correlation

            if ver >= 1:
                sys.stdout.write("Mutual Diffusion Coefficient Trajectory {} of {} complete\n".format(i + 1, Nmd))
        if ver >= 1:
            sys.stdout.write("\n")

        jrcf_mean = np.mean(jrcf, axis=2)

        # fit
        fitdcoeff = fit()
        ave_mutual_dcoeff = np.mean(mutual_dcoeff, axis=2)
        stddev_mutual_dcoeff = np.std(mutual_dcoeff, axis=2)

        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3, 5e-2, 2e3, 2e2]
            else:
                popt2 = [1e-4, 1e2]

        Value = np.zeros((ave_mutual_dcoeff.shape[0], ave_mutual_dcoeff.shape[1]))
        fitcurve = np.zeros_like(ave_mutual_dcoeff)
        fitcut = np.zeros_like(Value)
        for i in range(ave_mutual_dcoeff.shape[0]):
            for j in range(ave_mutual_dcoeff.shape[1]):
                Value[i, j], fitcurve[i, j, :], fitcut[i, j] = fitdcoeff.fit(
                    Time,
                    ave_mutual_dcoeff[i, j],
                    stddev_mutual_dcoeff[i, j],
                    use_double_exp,
                    popt2,
                    std_perc,
                    endt,
                )

        output = self.append_dict(
            Time,
            output,
            jrcf,
            jrcf_mean,
            mutual_dcoeff,
            ave_mutual_dcoeff,
            stddev_mutual_dcoeff,
            Value,
            fitcurve,
            fitcut,
        )

        return output

    def getmuldcoeff(
        self,
        vel: np.ndarray,
        Nskip: int,
        dt: float,
        nummoltype: np.ndarray,
        dump_frec: int,
        interval: int,
        molmass: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        molconc = nummoltype / nummoltype.sum()
        vel = vel[:, :, Nskip:]
        Ndim = vel.shape[0]
        Ndumps = vel.shape[2]
        Nmoltype = len(nummoltype)
        Ncorrelation = int((Nmoltype - 1) * (Nmoltype - 1))

        # Calculate particle current, eq.(3.4b) in Zhou J Phs Chem
        j = np.zeros((Nmoltype, Ndim, Ndumps))
        atomindex_start = 0
        atomindex_end = 0
        for typeindex, num in enumerate(nummoltype):
            atomindex_end += num
            j[typeindex, :, :] = vel[:, atomindex_start:atomindex_end, :].sum(axis=1)
            atomindex_start += num

        # Calculate relative particle current jr, eq.(4.4) in Zhou J Phs Chem
        # Though for better adaptability of the module, J_flux eq.(3.5) is calculated,
        # current module can only compute binary mixed systems
        jr = np.zeros((Nmoltype - 1, Ndim, Ndumps))
        J_flux = np.zeros((Nmoltype - 1, Ndim, Ndumps))
        m_bar = molmass @ molconc
        for i, m_alpha in enumerate(molmass[:-1]):
            for ii, m_beta in enumerate(molmass):
                delta_ab = 1 * (m_beta == m_alpha)
                jr[i, :, :] += (delta_ab - molconc[i]) * j[ii, :, :]
                J_flux[i, :, :] += (m_bar * delta_ab - molconc[i] * m_beta) * j[ii, :, :]
            J_flux[i, :, :] *= m_alpha / m_bar

        # Calculate the correlation function of jr in each direction
        correlation = np.zeros((Ncorrelation, Ndim + 1, Ndumps))
        indx = 0
        for i, jr1 in enumerate(jr):
            for ii, jr2 in enumerate(jr):
                for d in range(Ndim):
                    correlation[indx, d, :] = correlationfunction(jr1[d, :], jr2[d, :])
                    correlation[indx, -1, :] += correlation[indx, d, :]  # a dim for total correlation
            indx += 1

        # Calculate binary mutual diffusion coefficient, eq.(4.5) in Zhou J Phs Chem
        const = nummoltype.sum() * molconc.prod()
        Dt = dt * dump_frec * interval

        mutual_diffuso = cumulative_trapezoid(correlation, dx=Dt) / const / 1e8
        # m^2/s,  (A^2/ps) = 10^(-20)/10^(-12) = 1/(10^8)
        mutual_diffuso[:, -1, :] =  mutual_diffuso[:, -1, :] / 3 # for total
        Time = np.arange(mutual_diffuso.shape[2]) * Dt

        # The reason for not outputting J_flux is that it is not involved in the diffusion matrix calculation
        # in the binary system, the ternary part of the calculation is not written,
        # and the higher-order systems article does not give a specific formula for it
        return (Time, mutual_diffuso, correlation)

    def append_dict(
        self,
        Time: np.ndarray,
        output: Dict,
        jrcf: np.ndarray,
        jrcf_mean: np.ndarray,
        mutual_dcoeff: np.ndarray,
        ave_mutual_dcoeff: np.ndarray,
        stddev_mutual_dcoeff: np.ndarray,
        Value: np.ndarray,
        fitcurve: np.ndarray,
        fitcut: np.ndarray,
    ) -> Dict:
        if "D_m" not in output:
            output["D_m"] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        output["D_m"]["Units"] = "m^2/s"
        output["D_m"]["Time"] = Time.tolist()
        dim = ["x", "y", "z", "total"]
        for i in range(ave_mutual_dcoeff.shape[0]):
            for j in range(ave_mutual_dcoeff.shape[1]):
                output["D_m"]["JrCF"][str(i)][dim[j]] = jrcf[i, j].tolist()
                output["D_m"]["JrCF Average"][str(i)][dim[j]] = jrcf_mean[i, j].tolist()
                output["D_m"]["Integrals"][str(i)][dim[j]] = mutual_dcoeff[i, j].tolist()
                output["D_m"]["Average Value"][str(i)][dim[j]] = float(Value[i, j])
                output["D_m"]["Average Integral"][str(i)][dim[j]] = ave_mutual_dcoeff[i, j].tolist()
                output["D_m"]["Standard Deviation"][str(i)][dim[j]] = stddev_mutual_dcoeff[i, j].tolist()
                output["D_m"]["Fit"][str(i)][dim[j]] = fitcurve[i, j].tolist()
                output["D_m"]["Fit Cut"][str(i)][dim[j]] = int(fitcut[i, j])

        return output
