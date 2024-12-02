import numpy as np
import sys, copy
from collections import defaultdict
from scipy.integrate import cumtrapz
from OutputInfo import ReadBox, LammpsMDInfo
from Analysis.utils import correlationfunction
from Analysis.fit import fit


class MutualDcoeffJrcfCalc:

    def runMutualDcoeffjrcf(
        self,
        fileprefix,
        namemoltype,
        molmass,
        Nmd,
        Nskip,
        interval,
        use_double_exp,
        logname="log.lammps",
        velname="dump.vel",
        output={},
        popt2=None,
        endt=None,
        std_perc=None,
        ver=1,
    ):
        """
        This function calculates average and standard deviation of the mutual diffusion coefficient through
        relative particle current jr correlation function and fit the result with
        single or double-exponential function.

            Parameters:
                -----------
            fileprefix : str

            namemoltype : list of str
                A list of molecule labels corresponding to the `moltype` values, e.g., ['H', 'He'].

            molmass: list of int
                eg., for He and H system, [4,1]
                In fact, this item is not involved in the numerical computation of the diffusion
                matrix and only affects the diffusion fluxes

            Nmd : int
                num of md

            Nskip: int
                initial frames ignored during the calculation.
                Note that here is the number of frames skipped after reading the box,
                the interval parameter in "reading box" should be considered.

            interval: int
                reading interval of velosity dump data

            use_double_exp : bool
                weather use double-exponential fit

            logname : str, optional

            velname : str, optional

            output : dict, optional

            popt2 : list of float, optional
                initial guess value, if None, use [1e-4,1e2] for single-exponential fit,
                [1e-3,1.5e-1,1e2,1e3] for double-exponential fit

            endt ：float, optional
                cut time

            std_perc : float, optional
                "It was found empirically that the time at which the calculated standard deviation σ(t)
                was about 40% of the corresponding viscosity (rough average of the flat region in the running integral)
                was a good choice for tcut."
                https://pubs.acs.org/doi/10.1021/acs.jctc.5b00351.

                if endt=None, then use std_prec, if std_prec=None, then std_prec=0.4

            ver: int, optional
                if ver>1, output the progress
        """
        # read
        properties = ["vx", "vy", "vz"]

        if fileprefix == None:
            fileprefix = "./"

        logfilename = fileprefix + "000/" + logname
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        velfilename = fileprefix + "000/" + velname
        header, frames, atom_type = ReadBox.read_lammps_dump(velfilename, interval)
        print(len(frames))
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
            sys.stdout.write("Mutual Dcoeff Trajectory 1 of {} complete\n".format(Nmd))

        for i in range(1, Nmd):
            velfilename = fileprefix + str(i).zfill(3) + "/" + velname
            header, frames, atom_type = ReadBox.read_lammps_dump(velfilename, interval)
            steps, dumpinfo, bounds_matrices = zip(*frames)
            velosity = ReadBox.extract_dump_data(dumpinfo, properties)
            vel = velosity.transpose(2, 1, 0)

            (_, mutual_dcoeffo, correlation) = self.getmuldcoeff(
                vel, Nskip, dt, nummoltype, dump_frec, interval, molmass
            )

            if mutual_dcoeffo.shape[-1] < trjlen:
                trjlen = mutual_dcoeffo.shape[-1]

            mutual_dcoeff[:, :, i, :trjlen] = mutual_dcoeffo
            jrcf[:, :, i, : trjlen + 1] = correlation

            if ver >= 1:
                sys.stdout.write("Mutual Dcoeff Trajectory {} of {} complete\n".format(i + 1, Nmd))
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
        for i in range(ave_mutual_dcoeff.shape[0]):
            for j in range(ave_mutual_dcoeff.shape[1]):
                Value[i, j], fitcurve[i, j, :] = fitdcoeff.fit(
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
        )

        return output

    def getmuldcoeff(self, vel, Nskip, dt, nummoltype, dump_frec, interval, molmass):
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
        print(jr.shape)
        J_flux = np.zeros((Nmoltype - 1, Ndim, Ndumps))
        m_bar = molmass @ molconc
        for i, m_alpha in enumerate(molmass[:-1]):
            for ii, m_beta in enumerate(molmass):
                delta_ab = 1 * (m_beta == m_alpha)
                jr[i, :, :] += (delta_ab - molconc[i]) * j[ii, :, :]
                print(jr)
                J_flux[i, :, :] += (m_bar * delta_ab - molconc[i] * m_beta) * j[ii, :, :]
            J_flux[i, :, :] *= m_alpha / m_bar

        # Calculate the correlation function of jr in each direction
        correlation = np.zeros((Ncorrelation, Ndim + 1, Ndumps))
        indx = 0
        for i, jr1 in enumerate(jr):
            for ii, jr2 in enumerate(jr):
                for d in range(Ndim):
                    print(jr1[d, :], jr2[d, :])
                    print(jr)
                    correlation[indx, d, :] = correlationfunction(jr1[d, :], jr2[d, :])
                    correlation[indx, -1, :] += correlation[indx, d, :]  # a dim for total correlation
            indx += 1

        # Calculate binary mutual diffusion coefficient, eq.(4.5) in Zhou J Phs Chem
        const = 3 * nummoltype.sum() * molconc.prod()
        Dt = dt * dump_frec * interval
        mutual_diffuso = cumtrapz(correlation) * Dt / const / 1e8
        Time = np.arange(mutual_diffuso.shape[2]) * Dt

        # The reason for not outputting J_flux is that it is not involved in the diffusion matrix calculation
        # in the binary system, the ternary part of the calculation is not written,
        # and the higher-order systems article does not give a specific formula for it
        return (Time, mutual_diffuso, correlation)

    def append_dict(
        self,
        Time,
        output,
        jrcf,
        jrcf_mean,
        mutual_dcoeff,
        ave_mutual_dcoeff,
        stddev_mutual_dcoeff,
        Value,
        fitcurve,
    ):
        if "MutualDcoeffJrcf" not in output:
            output["MutualDcoeffJrcf"] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        output["MutualDcoeffJrcf"]["units"] = "m^2/s"
        output["MutualDcoeffJrcf"]["Time"] = Time
        dim = ["x", "y", "z", "total"]
        for i in range(ave_mutual_dcoeff.shape[0]):
            for j in range(ave_mutual_dcoeff.shape[1]):
                output["MutualDcoeffJrcf"]["Jrcf"][i][dim[j]] = copy.deepcopy(jrcf[i, j])
                output["MutualDcoeffJrcf"]["Jrcf Average"][i][dim[j]] = copy.deepcopy(jrcf_mean[i, j])
                output["MutualDcoeffJrcf"]["Integrals"][i][dim[j]] = copy.deepcopy(mutual_dcoeff[i, j])
                output["MutualDcoeffJrcf"]["Average Value"][i][dim[j]] = copy.deepcopy(Value[i, j])
                output["MutualDcoeffJrcf"]["Average Integral"][i][dim[j]] = copy.deepcopy(ave_mutual_dcoeff[i, j])
                output["MutualDcoeffJrcf"]["Standard Deviation"][i][dim[j]] = copy.deepcopy(stddev_mutual_dcoeff[i, j])
                output["MutualDcoeffJrcf"]["Fit"][i][dim[j]] = copy.deepcopy(fitcurve[i, j])
        return output
