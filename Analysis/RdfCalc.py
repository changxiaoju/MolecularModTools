import copy
import numpy as np


class RdfCalc:
    def runRdf(self, comx, comy, comz, Lx, Ly, Lz,
                  nummoltype, moltype, namemoltype, stabel_steps, binsize, 
                  maxr=None, output={}):
        
        """
        
        This function calculates the radial distribution function between the 
        center of mass for all species in the system
        
            Parameters:
                -----------

                comx, comy, comz : float
                    The x, y, and z coordinates of the center of mass, respectively.

                Lx, Ly, Lz : float
                    The dimensions of the simulation box in the x, y, and z directions.

                nummoltype :list of int
                    The number of different molecule types in the system. e.g., [20, 40]

                moltype : list of int
                    A list indicating the type of molecules in the system, e.g., [0, 1, 0, 0].

                namemoltype : list of str
                    A list of molecule labels corresponding to the `moltype` values, e.g., ['H', 'He'].

                stabel_steps : int
                    The number of frames to be used for the RDF calculation, selected after system relaxation. 
                    `firststep = numsteps - stabel_steps` ensures only stable configurations are considered.

                binsize : float
                    The size of the bins for calculating the radial distribution function.

                maxr : float, optional
                    The maximum radius to consider for the radial distribution function.
                    Defaults to `None`, meaning it will be calculated based on the system size.

                output : dict
                    A dictionary to store the results of the radial distribution analysis.
                    
        """
        (maxr, numbins, count, g,firststep) = self.setgparam(Lx, Ly, Lz, stabel_steps,
                                                            namemoltype, maxr,
                                                            binsize,len(comx))
        (count) = self.radialdistribution(g, len(comx[1]), moltype, comx,
                                              comy, comz, Lx, Ly, Lz, binsize,
                                              numbins, maxr, count)
        (radiuslist) = self.radialnormalization(numbins, binsize, Lx, Ly, Lz,
                                                nummoltype, count, g,
                                                firststep)
        self.append_dict(radiuslist, g, output, namemoltype)
        return output

    def setgparam(self, Lx, Ly, Lz, stabel_steps, namemoltype, maxr, binsize,numsteps):
        # uses side lengths to set the maximum radius for box and number of bins
        # also sets the first line using data on firststep and number of atoms
        firststep = numsteps-stabel_steps
        if maxr == None:
            maxr = min(Lx/2, Ly/2, Lz/2)
        else:
            maxr = float(maxr)
        numbins = int(np.ceil(maxr / binsize))
        count = firststep
        g = np.zeros((len(namemoltype), len(namemoltype), numbins))
        return maxr, numbins, count, g, firststep

    def radialdistribution(self, g, nummol, moltype, comx, comy, comz, Lx, Ly,
                           Lz, binsize, numbins, maxr, count):
        # calculates the number of molecules within each shell
        comxt = np.array(comx[count:]).transpose()
        comyt = np.array(comy[count:]).transpose()
        comzt = np.array(comz[count:]).transpose()
        indexlist = []
        #print(comxt)
        # change indeces order to com*[molecule][timestep]
        
        for i in range(0,len(g)):
            indexlist.append(np.array(moltype) == i)
            #creates two dimensional array
            #first dimension is molecule type
            #second dimension is over molecules
            #contains boolean for if that molecule is of the molecule type
        
        for molecule in range(0, nummol - 1):
            dx = comxt[molecule+1:] - np.tile(comxt[molecule],
                                              (len(comxt)-molecule-1,1))
            dy = comyt[molecule+1:] - np.tile(comyt[molecule],
                                              (len(comyt)-molecule-1,1))
            dz = comzt[molecule+1:] - np.tile(comzt[molecule],
                                              (len(comzt)-molecule-1,1))

            dx -= Lx * np.around(dx / Lx)
            dy -= Ly * np.around(dy / Ly)
            dz -= Lz * np.around(dz / Lz)
            #minimum image convention

            r2 = dx ** 2 + dy ** 2 + dz ** 2
            r = np.sqrt(r2)
            for i in range(0,len(indexlist)):
                gt,dist = np.histogram(r[indexlist[i][molecule+1:]].ravel(),
                                       bins=numbins,
                                       range=(0.5*binsize,binsize*(numbins+0.5)))
                g[moltype[molecule]][i]+= gt
                g[i][moltype[molecule]]+= gt
                
        count = len(comx)
        return count

    def radialnormalization(self, numbins, binsize, Lx, Ly, Lz, nummol, count,
                            g, firststep):
        # normalizes g to box density
        radiuslist = (np.arange(numbins) + 1) * binsize
        radiuslist = np.around(radiuslist,decimals = 3)
        for i in range(0, len(g)):
            for j in range(0, len(g)):
                g[i][j] *= Lx * Ly * Lz / nummol[i] / nummol[j] / 4 / np.pi / (
                               radiuslist) ** 2 / binsize / (
                               count - firststep)
        return radiuslist

    def append_dict(self, radiuslist, g, output, namemoltype):
        output['RDF'] = {}
        output['RDF']['units'] = 'unitless, angstroms'
        for i in range(0, len(namemoltype)):
            for j in range(i, len(namemoltype)):
                if not all([v == 0 for v in g[i][j]]):
                    output['RDF']['{0}-{1}'.format(namemoltype[i],
                                                   namemoltype[
                                                       j])] = copy.deepcopy(
                        g[i][j].tolist())
        if 'distance' not in list(output['RDF'].keys()):
            output['RDF']['distance'] = copy.deepcopy(radiuslist.tolist())
