import numpy as np
import sys,copy
from collections import defaultdict
from scipy.integrate import cumtrapz
from OutputInfo import ReadBox, LammpsMDInfo
from Analysis.utils import autocorrelate
from Analysis.fit import fit

class DcoeffVacfCalc():

    def runDcoeffVacf(self,fileprefix,namemoltype,Nmd,Nskip,interval,use_double_exp,logname='log.lammps',velname='dump.vel',output={},popt2=None,endt=None,std_perc=None,NCORES=1,ver=1):
        """
        This function calculates average and standard deviation of the diffusion coefficient through velosity autocorrelation function and fit the result with 
        single or double-exponential function.
        
            Parameters:
                -----------
            fileprefix : str

            namemoltype : list of str
                A list of molecule labels corresponding to the `moltype` values, e.g., ['H', 'He'].

            Nmd : int
                num of md

            Nskip: int
                initial lines ignored during the calculation

            interval: int
                reading interval of velosity dump data

            use_double_exp : bool
                weather use double-exponential fit
            
            logname : str, optional   

            velname : str, optional                
            
            output : dict, optional

            popt2 : list of float, optional
                initial guess value, if None, use [1e-4,1e2] for single-exponential fit, [1e-3,1.5e-1,1e2,1e3] for double-exponential fit

            endt ：float, optional
                cut time
            
            std_perc : float, optional
                "It was found empirically that the time at which the calculated standard deviation σ(t) was about 40% of the corresponding viscosity (rough average of the flat region in the running integral) was a good choice for tcut."
                https://pubs.acs.org/doi/10.1021/acs.jctc.5b00351.

                if endt=None, then use std_prec, if std_prec=None, then std_prec=0.4
            
            NCORES : int, optional
                should be smaller than core number
            
            ver: int, optional
                if ver>1, output the progress
        """
        #read
        properties = ['vx','vy','vz']

        if fileprefix == None:
                fileprefix = './'

        logfilename = fileprefix + '000/' + logname
        Nsteps, dt, dump_frec, thermo_frec = LammpsMDInfo.basic_info(logfilename)

        velfilename = fileprefix + '000/' + velname
        with open(velfilename, 'r') as lammpstrj_file:
            header, frames,atom_type = ReadBox.read_lammps_dump(lammpstrj_file,interval)
        steps, dumpinfo, bounds_matrices = zip(*frames)
        velosity = ReadBox.extract_dump_data(dumpinfo,properties)
        vel = velosity.transpose(2,1,0)
        nummoltype = np.unique(atom_type,return_counts=True)[1]

        #calculate
        (Time, dcoeffo, autocorrelation) = self.getdcoeff(vel, Nskip, dt, nummoltype, dump_frec, interval, NCORES)

        trjlen = len(Time)
        Ndim = vel.shape[0]
        Nmoltype = len(nummoltype)

        dcoeff = np.zeros((Nmoltype,Ndim+1,Nmd,trjlen))
        vacf = np.zeros((Nmoltype,Ndim+1,Nmd,trjlen+1))
        dcoeff[:, :, 0, :] = dcoeffo
        vacf[:, :, 0, :] = autocorrelation

        if ver>=1:
            sys.stdout.write('Dcoeff Trajectory 1 of {} complete\n'.format(Nmd))

        for i in range(1,Nmd):
            velfilename = fileprefix+str(i).zfill(3) + '/' + velname
            with open(velfilename, 'r') as lammpstrj_file:
                header, frames,atom_type = ReadBox.read_lammps_dump(lammpstrj_file,interval)
            steps, dumpinfo, bounds_matrices = zip(*frames)
            velosity = ReadBox.extract_dump_data(dumpinfo,properties)
            vel = velosity.transpose(2,1,0)

            (_, dcoeffo, autocorrelation) = self.getdcoeff(vel, Nskip, dt, nummoltype, dump_frec, interval, NCORES)

            if dcoeffo.shape[-1] < trjlen:
                trjlen = dcoeffo.shape[-1]
                
            dcoeff[:, :, i, :trjlen] = dcoeffo
            vacf[:, :, i, :trjlen+1] = autocorrelation

            if ver>=1:
                sys.stdout.write('Dcoeff Trajectory {} of {} complete\n'.format(i+1,Nmd))
        if ver>=1:
            sys.stdout.write('\n')

        vacf_mean = np.mean(vacf,axis=2)


        #fit
        fitdcoeff = fit()
        ave_dcoeff = np.mean(dcoeff,axis=2)
        stddev_dcoeff = np.std(dcoeff,axis=2)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3,5e-2,2e3,2e2]
            else:
                popt2 = [1e-4,1e2]

        Value = np.zeros((ave_dcoeff.shape[0],ave_dcoeff.shape[1]))
        fitcurve = np.zeros_like(ave_dcoeff)
        for i in range(ave_dcoeff.shape[0]):
            for j in range(ave_dcoeff.shape[1]):
                Value[i, j],fitcurve[i, j, :] = fitdcoeff.fit(Time,ave_dcoeff[i, j],stddev_dcoeff[i, j],use_double_exp,popt2,std_perc=None,endt=None)
        
        output = self.append_dict(Time,namemoltype, output, vacf, vacf_mean, dcoeff, ave_dcoeff, stddev_dcoeff, Value, fitcurve)
        return output
    
    def getdcoeff(self, vel, Nskip, dt, nummoltype, dump_frec, interval, NCORES):

        p=Pool(NCORES)

        vel = vel[:,:,Nskip:]
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
                    atomindex_autocorrelation = autocorrelate(vel[d, atomindex, :])

                    tmp_autocorrelation += atomindex_autocorrelation

                autocorrelation[typeindex, d, :] = tmp_autocorrelation / num
                autocorrelation[typeindex, -1, :] += tmp_autocorrelation / num
                atomindex_start += num
                
        diffuso = cumtrapz(autocorrelation)*dt/300000000
        Time = np.arange(diffuso.shape[2])*dt*dump_frec*interval
        p.close()

        return (Time, diffuso, autocorrelation)
    
    def append_dict(self,Time,namemoltype, output, vacf, vacf_mean, dcoeff, ave_dcoeff, stddev_dcoeff, Value, fitcurve):
        if 'DcoeffVacf' not in output:
            output['DcoeffVacf'] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        output['DcoeffVacf']['units'] = 'm^2/s'
        output['DcoeffVacf']['Time']=Time
        dim = ['x', 'y', 'z', 'total']
        for i in range(ave_dcoeff.shape[0]):
            for j in range(ave_dcoeff.shape[1]):
                output['DcoeffVacf']['vacf'][namemoltype[i]][dim[j]] = copy.deepcopy(vacf[i, j])
                output['DcoeffVacf']['vacf Average'][namemoltype[i]][dim[j]] = copy.deepcopy(vacf_mean[i, j])
                output['DcoeffVacf']['Integrals'][namemoltype[i]][dim[j]] = copy.deepcopy(dcoeff[i, j])
                output['DcoeffVacf']['Average Value'][namemoltype[i]][dim[j]] = copy.deepcopy(Value[i, j])
                output['DcoeffVacf']['Average Integral'][namemoltype[i]][dim[j]] = copy.deepcopy(ave_dcoeff[i, j])
                output['DcoeffVacf']['Standard Deviation'][namemoltype[i]][dim[j]] = copy.deepcopy(stddev_dcoeff[i, j])
                output['DcoeffVacf']['Fit'][namemoltype[i]][dim[j]] = copy.deepcopy(fitcurve[i, j])
        return output
    
    
    
