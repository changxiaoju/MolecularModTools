import sys
import numpy as np
from multiprocessing import Pool
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.integrate import cumtrapz
from Analysis.utils import autocorrelate
from Analysis.fit import fit

from OutputInfo import LammpsMDInfo


class ViscCalc():

    def runVisc(self,logfileprefix,logname,nummd,numskip,use_double_exp,output={},popt2=None,endt=None,std_perc=None,NCORES=1,ver=1): 
        """
        This function calculates average and standard deviation of the viscosity and fit the result with 
        single or double-exponential function.
        
            Parameters:
                -----------
            thermo_df : DataFrame
                thermo dataframe read from log file, to compute viscosity, it should contain pressure tensor 

            logfileprefix : str

            logname : str                

            nummd : int
                num of md

            numskip: int
                initial lines ignored during the calculation

            use_double_exp : bool
                weather use double-exponential fit
            
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
        # read
        output['Viscosity']={}
        output['Viscosity']['Units']='mcP'
        if logfileprefix==None:
            logfileprefix='./'
        logfilename=logfileprefix+'000/'+logname
        thermo_df = LammpsMDInfo.read_thermo(logfilename)
        Nsteps, trj_dump, thermo_dump, dt = LammpsMDInfo.basic_info(logfilename)

        # calculate
        (Time,visco,pcorr)=self.getvisc(thermo_df, numskip, dt, NCORES)
        trjlen = len(Time)
        viscosity = np.zeros((nummd,trjlen))
        autocorrelation = np.zeros((nummd,trjlen+1))
        viscosity[0] = visco 
        autocorrelation[0] = pcorr
        if ver>=1:
            sys.stdout.write('Viscosity Trajectory 1 of {} complete\n'.format(nummd))
        
        for i in range(1,nummd):
            logfilename=logfileprefix+str(i).zfill(3)+'/'+logname
            thermo_df = LammpsMDInfo.read_thermo(logfilename)
            (Time,visco,pcorr)=self.getvisc(thermo_df, numskip,dt, NCORES)
            if len(visco) < trjlen:
                trjlen = len(visco)
            viscosity[i, :trjlen] = visco
            autocorrelation[i, :trjlen+1] = pcorr
            if ver>=1:
                sys.stdout.write('\rViscosity Trajectory {} of {} complete\n'.format(i+1,nummd))
        if ver>=1:
            sys.stdout.write('\n')

        output['Time']=Time[:trjlen]
        acf_mean = np.mean(autocorrelation,axis=0)
        output['Autocorrelations'] = autocorrelation
        output['Autocorrelation Average'] = acf_mean

        # fit
        fitvisc = fit()
        ave_visc = np.mean(viscosity,axis=0)
        stddev_visc = np.std(viscosity,axis=0)
        if popt2 is None:
            if use_double_exp:
                popt2 = [2e-3,5e-2,2e3,2e2]
            else:
                popt2 = [1e-4,1e2]
        Value,fitvalue = fitvisc.fit(Time,ave_visc,stddev_visc,use_double_exp,popt2,std_perc,endt)

        output['Viscosity Integrals'] = viscosity
        output['Viscosity Average Value'] = Value
        output['Viscosity Average Integral'] = ave_visc
        output['Viscosity Standard Deviation'] = stddev_visc
        output['Viscosity Fit'] = fitvalue

        return(output)

    def getvisc(self, thermo_df, numskip, dt, NCORES):

        p=Pool(NCORES)

        numtimesteps = len(thermo_df['Pxy'])
        a1=thermo_df['Pxy'][numskip:]
        a2=thermo_df['Pxz'][numskip:]
        a3=thermo_df['Pyz'][numskip:]
        a4=thermo_df['Pxx'][numskip:]-thermo_df['Pyy'][numskip:]
        a5=thermo_df['Pyy'][numskip:]-thermo_df['Pzz'][numskip:]
        a6=thermo_df['Pxx'][numskip:]-thermo_df['Pzz'][numskip:]
        array_array=[a1,a2,a3,a4,a5,a6]
        pv=p.map(autocorrelate,array_array)
        pcorr = (pv[0]+pv[1]+pv[2])/6+(pv[3]+pv[4]+pv[5])/12  

        temp=np.mean(thermo_df['Temp'][numskip:])

        visco = (cumtrapz(pcorr,thermo_df['Step'][:len(pcorr)]))*dt*10**-12*1000*101325.**2*thermo_df['Volume'].iloc[-1]*10**-30/(1.38*10**-23*temp)
        Time = np.array(thermo_df['Step'][:len(pcorr)-1])*dt
        p.close()

        return (Time,visco,pcorr)


