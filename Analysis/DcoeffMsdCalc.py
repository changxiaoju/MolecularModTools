import numpy as np
from scipy import stats
import warnings

class DcoeffMsdCalc:
    
    def runDcoeffMsd(self, namemoltype, dt, tol, output):
        
        """
        This function fits the mean square displacement to calculate the
        diffusivity for all molecule types in the system
        """
        
        output['Diffusivity'] = {}
        output['Diffusivity']['units'] = 'm^2/s'
        for i in range(0,len(namemoltype)):
            MSD = output['MSD'][namemoltype[i]]
            lnMSD = np.log(MSD[1:])
            time = output['MSD']['time']
            lntime = np.log(time[1:])
            firststep = self.findlinearregion(lnMSD, lntime, dt, tol)
            diffusivity = self.getdiffusivity(time, MSD, firststep)
            output['Diffusivity'][namemoltype[i]] = diffusivity
            
    
    
    def findlinearregion(self, lnMSD, lntime, dt, tol):
        #Uses the slope of the log-log plot to find linear regoin of MSD
        timestepskip=np.ceil(1/dt)
        linearregion=True
        maxtime = len(lnMSD)
        numskip=1
        while linearregion == True:
            if numskip*timestepskip+1 > maxtime:
                firststep = maxtime-1-(numskip-1)*timestepskip
                return firststep
                linearregion= False
            else:
                t1=int(maxtime-1-(numskip-1)*timestepskip)
                t2=int(maxtime-1-numskip*timestepskip)
                slope = (lnMSD[t1]-lnMSD[t2])/(lntime[t1]-lntime[t2])
                if abs(slope-1.) < tol:
                    numskip += 1
                else:
                    firststep=t1
                    return firststep
                    linearregion = False
                
                
    def getdiffusivity(self, Time, MSD, firststep):
        #Fits the linear region of the MSD to obtain the diffusivity
        calctime = []        
        calcMSD = []
        for i in range(int(firststep), len(Time)):
            calctime.append(Time[i])
            calcMSD.append(MSD[i])
        if len(calctime)==1:
            diffusivity = 'runtime not long enough'
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                line = stats.linregress(calctime,calcMSD)
            slope = line[0]
            diffusivity=slope/600000000 # m^2/s,  (A^2/ps)/6 = 10^(-20)/10^(-12)/6 = 1/(6*10^8)
        return diffusivity
            
