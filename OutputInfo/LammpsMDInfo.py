import pandas as pd
import numpy as np
import re,sys
from io import StringIO

def basic_info(log_file):

    #numlines = int(sum(1 for line in open(log_file)))
    numlines = 500 # usually, 500 lines is enough for input information in log file.
    logfile = open(log_file)
    foundrunstep, foundtrjdump, foundthermodump, foundtimestep = False, False, False, False
    i = 0
    while not (foundrunstep and foundtrjdump and foundthermodump and foundtimestep):
        if i > numlines:
            missing_info = []
            if not foundrunstep:
                missing_info.append('run step')
                Nsteps = None
            if not foundtrjdump:
                missing_info.append('trj dump')
                trj_dump = None
            if not foundthermodump:
                missing_info.append('thermo dump')
                thermo_dump = None
            if not foundtimestep:
                missing_info.append('timestep')
                dt = None
            print('could not decide the following: ' + ', '.join(missing_info))
            return Nsteps, dt, trj_dump, thermo_dump

        line = logfile.readline()
        line = line.split()
        if len(line) > 1:
            if line[0] == 'run':
                try:
                    Nsteps = int(line[1])
                    foundrunstep = True
                except:
                    pass
            if line[0] == 'dump' and any(keyword in line for keyword in ['x', 'y', 'z', 'xs', 'ys', 'zs', 'atom']): # more complicated output format is ignored.
                try:
                    trj_dump = int(line[4])
                    foundtrjdump = True
                except:
                    pass
            if line[0] == 'thermo' and line[1].isdigit():
                try:
                    thermo_dump = int(line[1])
                    foundthermodump = True
                except:
                    pass
            if line[0] == 'timestep':
                try:
                    dt = float(line[1])
                    foundtimestep=True
                except:
                    pass
        i += 1
    logfile.close()
    return Nsteps, dt, trj_dump, thermo_dump

def thermo_info(log_file):
    """
    
    Parameters
    ----------
    log_file: string
        Name of lammps MD log file
        
    Returns
    -------
    df: DataFrame
        The dataframe with header like "Step Temp PotEng..."
    
    """
    with open(log_file, 'r') as file:
        data_started = False
        data_lines = []
        header = []

        for line in file:
            # Check if the line starts with "Step", indicating the start of the data section
            if 'Step' in line:
                data_started = True
                header = line.split()  # Save the header
                continue

            # If data has started, collect lines that contain numerical data
            if data_started:
                if line.strip():  # Make sure line is not empty
                    data_lines.append(line)

                # Break out of the loop if a blank line is encountered, indicating the end of the table
                elif not line.strip():
                    break
    # Convert the data into a numpy array
    data_array = np.genfromtxt(StringIO("\n".join(data_lines)), dtype=float)

    # Optionally convert to a DataFrame if necessary
    df = pd.DataFrame(data_array, columns=header)
    df = df.dropna()
    return df

