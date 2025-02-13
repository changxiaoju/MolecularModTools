# Note:
1. The default input unit for this project is the "metal" unit in LAMMPS. If other units such as "real" are used during the MD, the units need to be changed in the corresponding .py files.
2. When calculating thermal conductivity, viscosity and diffusion coefficient, it is necessary to average multiple independent MD simulations. By default, we assume that the file names of these directories are "case.*", and use the following Python code to generate the input files for these independent MD simulations:
```python
import os
import random

 # the prepare folder should conclude input file(input.lammps), datafile(conf.lmp), model file(frozen_model.pb), submit file(job.jsaon)
samplefile = open('prepare/input.lammps','r').readlines() 
 # number of independent MD simulations
filelist= range(0,5) 
for num in filelist:
    os.system('mkdir case.{0}'.format(str(num).zfill(3)))
    output = open('case.{0}/input.lammps'.format(str(num).zfill(3)),'w')
    for line in range(0,len(samplefile)):
        # The line number where the initial velocity is located
        if line == 32:
            output.write('velocity        all create  ${{TEMP}} {0}  \n'.format(random.randint(1,999999999)))
        else:
            output.write(samplefile[line])
    output.close()
    os.system('cp prepare/conf.lmp case.{0}/'.format(str(num).zfill(3)))
    os.system('cp prepare/frozen_model.pb case.{0}/'.format(str(num).zfill(3)))
    os.system('cp prepare/job.json case.{0}/'.format(str(num).zfill(3)))
```
3. example.ipynb will be uploaded later.
