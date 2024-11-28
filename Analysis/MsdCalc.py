import numpy as np
import os
import time
import copy
import warnings
import sys


class MsdCalc:

    def runMsd(
        self,
        comx,
        comy,
        comz,
        Lx,
        Ly,
        Lz,
        moltype,
        namemoltype,
        dt,
        skip,
        num_init=None,
        ver=True,
        output={},
    ):
        """
        This function calculates the mean square displacement for all molecule
        types in the system from center of mass positions

         Parameters:
            -----------
            moltype : list of int
                A list indicating the type of molecules in the system, e.g., [0, 1, 0, 0].

            namemoltype : list of str
                A list of molecule labels corresponding to the `moltype` values, e.g., ['H', 'He'].

            dt : float
                The timestep of the trajectory file in units of time.

            skip : int
                The number of initial frames to skip during analysis.

            num_init : int, optional
                The number of initial timesteps for MSD calculation.
                By default, it is half of the trajectory after removing the skipped frames.

            ver : bool, optional
                A flag to indicate whether to print progress during computation.

            output : dict, optional
                A dictionary to store the results, e.g., output = {}.
        """
        Lx2, Ly2, Lz2 = Lx / 2, Ly / 2, Lz / 2
        (comx, comy, comz) = self.unwrap(comx, comy, comz, Lx, Ly, Lz, Lx2, Ly2, Lz2)
        if ver > 0:
            print("unwrap complete")
        num_timesteps = len(comx)
        (num_init, len_MSD, MSD, diffusivity) = self.gettimesteps(num_timesteps, namemoltype, skip, num_init)
        (molcheck, nummol) = self.setmolarray(moltype, namemoltype)
        for i in range(skip, num_init + skip):
            for j in range(i, i + len_MSD):
                r2 = self.calcr2(comx, comy, comz, i, j)
                MSD = self.MSDadd(r2, MSD, molcheck, i, j)
            if ver:
                sys.stdout.write("\rMSD calculation {:.2f}% complete".format((i + 1 - skip) * 100.0 / num_init))
        if ver:
            sys.stdout.write("\n")
        MSD = self.MSDnorm(MSD, num_init, nummol)
        Time = self.createtime(dt, len_MSD)
        self.append_dict(MSD, namemoltype, output, Time)
        return output

    def unwrap(self, comx, comy, comz, Lx, Ly, Lz, Lx2, Ly2, Lz2):
        # unwraps the coordintes of the molecules
        # assumes if a molecule is more than half a box length away from its
        # previous coordinte that it passed through a periodic boundary
        for i in range(1, len(comx)):
            for j in range(0, len(comx[i])):
                if (comx[i][j] - comx[i - 1][j]) > Lx2:
                    while (comx[i][j] - comx[i - 1][j]) > Lx2:
                        comx[i][j] -= Lx
                elif (comx[i][j] - comx[i - 1][j]) < (-Lx2):
                    while (comx[i][j] - comx[i - 1][j]) < (-Lx2):
                        comx[i][j] += Lx

                if (comy[i][j] - comy[i - 1][j]) > Ly2:
                    while (comy[i][j] - comy[i - 1][j]) > Ly2:
                        comy[i][j] -= Ly
                elif (comy[i][j] - comy[i - 1][j]) < (-Ly2):
                    while (comy[i][j] - comy[i - 1][j]) < (-Ly2):
                        comy[i][j] += Ly

                if (comz[i][j] - comz[i - 1][j]) > Lz2:
                    while (comz[i][j] - comz[i - 1][j]) > Lz2:
                        comz[i][j] -= Lz
                elif (comz[i][j] - comz[i - 1][j]) < (-Lz2):
                    while (comz[i][j] - comz[i - 1][j]) < (-Lz2):
                        comz[i][j] += Lz
        return (comx, comy, comz)

    def gettimesteps(self, num_timesteps, namemoltype, skip, num_init=None):
        # Calculates the length of the trajectory
        # Uses length to determine length of MSD and number of initial timesteps
        if num_init == None:
            num_init = int(np.floor((num_timesteps - skip) / 2))
        else:
            num_init = int(num_init)

        len_MSD = num_timesteps - skip - num_init
        MSD = np.zeros((len(namemoltype), len_MSD))
        diffusivity = []
        return (num_init, len_MSD, MSD, diffusivity)

    def setmolarray(self, moltype, namemoltype):
        # Generates arrays for dot product calculation
        # Array is MxN where M is number of molecule types and N is number of molecules
        # value is 1 if molecule N is of type M else is 0
        molcheck = np.zeros((len(namemoltype), len(moltype)))
        for i in range(0, len(moltype)):
            molcheck[moltype[i]][i] = 1
        nummol = np.zeros(len(namemoltype))
        for i in range(0, len(nummol)):
            nummol[i] = np.sum(molcheck[i])
        return (molcheck, nummol)

        """
        e.g.,
        moltype = [0,1,0,0]
        namemoltype = ['H','He']
        
        molcheck:
        array([[1., 0., 1., 1.],
               [0., 1., 0., 0.]])
        """

    def calcr2(self, comx, comy, comz, i, j):
        # Calculates distance molecule has traveled between steps i and j
        r2 = (comx[j] - comx[i]) ** 2 + (comy[j] - comy[i]) ** 2 + (comz[j] - comz[i]) ** 2
        return r2

    def MSDadd(self, r2, MSD, molcheck, i, j):
        # Uses dot product to calculate average MSD for a molecule type
        for k in range(0, len(molcheck)):
            sr2 = np.dot(r2, molcheck[k])
            MSD[k][j - i] += sr2
        return MSD

    def MSDnorm(self, MSD, MSDt, nummol):
        # Normalize the MSD by number of molecules and number of initial timesteps
        for i in range(0, len(nummol)):
            MSD[i] /= MSDt * nummol[i]

        return MSD

    def createtime(self, dt, MSDt):
        # Creates an array of time values
        Time = np.arange(0, MSDt, dtype=float)
        Time *= dt
        return Time

    def append_dict(self, MSD, namemoltype, output, Time):
        # Write MSD to output dictionary
        output["MSD"] = {}
        output["MSD"]["units"] = "Angstroms^2, ps"
        for i in range(0, len(namemoltype)):
            output["MSD"][namemoltype[i]] = copy.deepcopy(MSD[i].tolist())

        output["MSD"]["time"] = copy.deepcopy(Time.tolist())
