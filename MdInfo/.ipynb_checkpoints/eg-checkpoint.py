from Constant import *
from Dataset import _get_EOS

from monty.serialization import loadfn,dumpfn

class DPGenSys():
    
    # ========================================= #
    # read basic information for DPGEN run
    # dpgen_dir : the working directory of DPGEN
    # ========================================= #
    def __init__(self, dpgen_dir, printf = True):
        
        self.dir = dpgen_dir
        self.param_file = os.path.join(dpgen_dir, 'param.json')
        self.jdata =loadfn(self.param_file)
        
        #self.N_iter = len(glob.glob( os.path.join( dpgen_dir ,'iter.*' )))
        self.N_iter = len(self.jdata['model_devi_jobs'])
        
        self.confs_list = self.jdata['sys_configs']
        self.label_list = []
        
        self.printf = printf
        
        print("DPGenerator System contains %.d Iterations"%(self.N_iter) )
        print("There are %.d initial configurations for exploration: \t "%(len(self.confs_list)) )
        for cc in self.confs_list:
            label = cc[0].split('.')[0]
            print(label,'\t\t')
            self.label_list.append(label)

    
    def _learning_curve(self, iter_idx, model_idx=0):
        
        iter_idx = 'iter.%.6d'%iter_idx
        model_idx = '%.3d'%model_idx
    
        path = os.path.join(self.dir,iter_idx, '00.train', model_idx, 'lcurve.out')

        return np.loadtxt(path)
        
    def _model_devi(self, iter_idx, sys_idx = 0, case_idx = 0):
        
        iter_idx = 'iter.%.6d'%iter_idx
        task_idx = 'task.%.3d'%sys_idx+'.%.6d'%case_idx  
        
        path = os.path.join(self.dir,iter_idx, '01.model_devi', task_idx, 'model_devi.out')

        return np.loadtxt(path)

    def _get_PT(self, iter_idx, sys_idx = 0, case_idx = 0):
        
        iter_idx = 'iter.%.6d'%iter_idx
        task_idx = 'task.%.3d'%sys_idx+'.%.6d'%case_idx  
        
        path = os.path.join(self.dir,iter_idx, '01.model_devi', task_idx, 'input.lammps')

        temp = float(os.popen("grep ' TEMP ' " + path).readlines()[0].split()[-1])
        ele_temp = float(os.popen("grep ' ELE_TEMP ' " + path).readlines()[0].split()[-1])
        press = float(os.popen("grep ' PRES ' " + path).readlines()[0].split()[-1]) * bar2Pa /1e9
        return temp, ele_temp, press, 
    
    def _sampling(self, iter_idx, sys_idx = 0):
        
        iter_idx = 'iter.%.6d'%iter_idx
        data_idx = 'data.%.3d'%sys_idx
        
        path = os.path.join(self.dir, iter_idx, '02.fp', data_idx)
        
        # return (temps, press, vol, energy)
        return _get_EOS(path)
    
    def _plot_lc(self, ax, iter_idx, model_idx=0):
        #fig, ax = plt.subplots(1,3, figsize=(8,2),dpi=200)

        data = self._learning_curve(iter_idx, model_idx)

        ll_list = ['Energy (eV)','Force (eV/$\\rm{\mathring A}$)','Virial (eV)']
        for idx in range(3):
            if idx == 0:
                ax[idx].plot(data[:,0],data[:,2+idx],'o',ms=1,mew=0.5, label='Iter.%.3d (DP%.3d)'%(iter_idx, model_idx))
            else:
                ax[idx].plot(data[:,0],data[:,2+idx],'o',ms=1,mew=0.5)

            ax[idx].set_xscale('log')
            ax[idx].set_yscale('log')
            ax[idx].set_title(ll_list[idx])
            ax[idx].set_xlabel('stopbatch')
            
    def _plot_md(self, ax, iter_idx, sys_idx = 0, case_idx = 0, show_ele_temp = False):
        
        lo= self.jdata['model_devi_f_trust_lo']
        hi= self.jdata['model_devi_f_trust_hi']

        data = self._model_devi(iter_idx, sys_idx , case_idx)
        
        
        ax.plot(data[:,0],data[:,4],'o', mfc='none',mew=0.5,ms=2)
        
        ax.axhline(lo,linestyle='--',lw=0.5,color='k')
        ax.axhline(hi,linestyle='--',lw=0.5,color='k')  
        
        try:
            t,te,p = self._get_PT(iter_idx, sys_idx , case_idx)

            output = 'T_i = %.d K'%t
            if show_ele_temp:
                output += 'T_e = %.d K'%te
            output += ' p = %.2f GPa, '%p
            output += self.label_list[sys_idx]
            ax.set_title(output, fontsize=8)
        except:
            pass

        ax.set_ylabel('model_devi ($\\rm{eV/\mathring A}$)')
        ax.set_xlabel('timestep')
        ax.set_xlim(data[0,0],data[-1,0])
        
    def _plot_sampling(self, ax, iter_idx,  sys_idx = 0, color='dodgerblue', is_label=False, label=''):
        
        self.temps, self.press, self.vol, self.energy, self.natoms = self._sampling(iter_idx, sys_idx)

        if is_label:
            ax.plot(self.press, self.temps, 
                'o', color=color, alpha=0.4, mew=0.5, mfc='none',ms=3, label=label)   
        else:
            ax.plot(self.press, self.temps, 
                'o', color=color, alpha=0.4, mew=0.5, mfc='none',ms=3)   
        
        if self.printf:
            output = 'Iter %.6d '%iter_idx
            output += ' add %.d frames '%self.temps.shape[0]
            output += '-- sys.%.3d '%sys_idx 
            output += '(' + self.label_list[sys_idx] +')'

            print(output)

        ax.set_xlabel('Pressure (GPa)')
        ax.set_ylabel('Temperature (K)')


    def _all_sampling_plot(self, ax, color):

        count = np.zeros(len(self.confs_list))
        
        self.frame_sys = np.zeros(len(self.confs_list))
        
        for i in range(self.N_iter):
            
            for sys_idx in range(len(self.confs_list)):
                
                try:
                    # label only plots in the first time for single system
                    if count[sys_idx] == 0:
                        self._plot_sampling(ax, iter_idx = i, sys_idx = sys_idx, 
                                           color=color[sys_idx], is_label=True, 
                                            label='DPGEN ('+self.label_list[sys_idx]+')')
                        count[sys_idx] +=1 
                    else:
                        self._plot_sampling(ax, iter_idx = i, sys_idx = sys_idx, 
                                           color=color[sys_idx])
                
                    self.frame_sys[sys_idx] += self.temps.shape[0]
                
                    if self.printf:
                        output = 'Iter %.6d --- sys. %.3d'%(i, sys_idx)
                        output += ' %.d frames in total '%self.frame_sys[sys_idx]
                        
                        print(output)
                
                except:
                    pass        

    def _get_extra(self, iter_idx, sys_idx = 0, str_s='energy  without entropy', save_raw = 'internal.raw'):
        
        iter_idx = 'iter.%.6d'%iter_idx

        path = os.path.join(self.dir, iter_idx, '02.fp', 'task.%.3d.*'%sys_idx, 'OUTCAR')

        fp_list = glob.glob( path )

        tmp = []
        
        for task_idx in range(len(fp_list)):
            
            outcar = os.path.join(self.dir, iter_idx, '02.fp', 'task.%.3d.%.6d'%(sys_idx, task_idx), 'OUTCAR')
        
            #print(outcar)
        
            cmd = "grep '"+str_s+"' " + outcar
           
            tmp = np.append(tmp, float(os.popen(cmd).readlines()[0].split()[3]) )
            
        save_dir = os.path.join(self.dir, iter_idx, '02.fp', 'data.%.3d'%sys_idx)
        np.savetxt( os.path.join(save_dir, save_raw), tmp.reshape(-1,1) )
    
        print(os.path.join(save_dir, save_raw),' is generated')
        
        
    def _obtain_tt_te_sampling(self):
        
        input_list = glob.glob( os.path.join(self.dir, 'iter.*', '01.model_devi', 'task*', 'input.lammps') )

        tt = []
        te = []

        for path in input_list:
            temp = float(os.popen("grep ' TEMP ' " + path).readlines()[0].split()[-1])
            ele_temp = float(os.popen("grep ' ELE_TEMP ' " + path).readlines()[0].split()[-1])
            #press = float(os.popen("grep ' PRES ' " + path).readlines()[0].split()[-1]) * bar2Pa /1e9

            tt = np.append(tt, temp)
            te = np.append(te, ele_temp)
            
        return tt,te
        
                
    def _collect_data_to(self, out_dir, set_numb = 20000, prefix=''):
        
        
        for sys_idx in range(len(self.label_list)):

            
            file_list = glob.glob(os.path.join(self.dir, 'iter.*', '02.fp', prefix+'data.%.3d'%sys_idx))
            file_list.sort()
            
            print('Sys.%.3d is working, there are %.d sub-datasets '%(sys_idx, len(file_list)))
            
            if len(file_list) != 0:

                fparam = []
                internal = []
                ms = dpdata.MultiSystems()

                for file in file_list:

                    sys = dpdata.LabeledSystem( file, fmt='deepmd/raw' )
                    ms.append(sys)

                    try:
                        tmp = np.loadtxt( os.path.join(file,'fparam.raw') )
                        fparam = np.append(fparam, tmp)
                        
                        is_fparam = True
                        
                    except:
                        is_fparam = False
                        
                    try:
                        tmp    = np.loadtxt( os.path.join(file,'internal.raw') )
                        internal = np.append(internal, tmp)
                        
                        is_internal = True
                    except:
                        is_internal = False                        

                outdir_ss = os.path.join(out_dir, self.label_list[sys_idx])

                if not os.path.exists(outdir_ss):
                    os.mkdir( outdir_ss)

                natom = sys.get_natoms()
                aparam = np.expand_dims(fparam, 0).repeat(natom,axis=0).T
                    
                ms.to_deepmd_raw(outdir_ss)
                
                if is_fparam:
                    np.savetxt( os.path.join(outdir_ss, 'fparam.raw'), fparam)
                    np.savetxt( os.path.join(outdir_ss, 'aparam.raw'), aparam)
                
                    print('NOTE: fparam.raw is generated')
                if is_internal:
                    np.savetxt( os.path.join(outdir_ss, 'internal.raw'), internal)
                    print('NOTE: internal.raw is generated')
                    
                os.chdir(outdir_ss)
                os.system('mv ./*/*.raw ./')
                
                os.system('~/raw_to_set.sh %.d'%(set_numb))

            print('Sys.%.3d is done'%(sys_idx))

    #def _extract_internal_energy(self):
        
        
        