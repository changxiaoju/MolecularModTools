import pandas as pd
import numpy as np
import re, sys
from io import StringIO


def basic_info(log_file):

    # numlines = int(sum(1 for line in open(log_file)))
    numlines = 500  # usually, 500 lines is enough for input information in log file.
    logfile = open(log_file)
    foundrunstep, foundtrjdump, foundthermodump, foundtimestep = False, False, False, False
    i = 0
    while not (foundrunstep and foundtrjdump and foundthermodump and foundtimestep):
        if i > numlines:
            missing_info = []
            if not foundrunstep:
                missing_info.append("run step")
                Nsteps = None
            if not foundtrjdump:
                missing_info.append("trj dump")
                dump_frec = None
            if not foundthermodump:
                missing_info.append("thermo dump")
                thermo_frec = None
            if not foundtimestep:
                missing_info.append("timestep")
                dt = None
            print("could not decide the following: " + ", ".join(missing_info))
            return Nsteps, dt, dump_frec, thermo_frec

        line = logfile.readline()
        line = line.split()
        if len(line) > 1:
            if line[0] == "run":
                try:
                    Nsteps = int(line[1])
                    foundrunstep = True
                except:
                    pass
            if line[0] == "dump" and any(
                keyword in line for keyword in ["x", "y", "z", "xs", "ys", "zs", "vx", "vy", "vz", "atom"]
            ):  # more complicated output format is ignored.
                try:
                    dump_frec = int(line[4])
                    foundtrjdump = True
                except:
                    pass
            if line[0] == "thermo" and line[1].isdigit():
                try:
                    thermo_frec = int(line[1])
                    foundthermodump = True
                except:
                    pass
            if line[0] == "timestep":
                try:
                    dt = float(line[1])
                    foundtimestep = True
                except:
                    pass
        i += 1
    logfile.close()
    return Nsteps, dt, dump_frec, thermo_frec


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
    df_list = []

    with open(log_file, "r") as file:

        for line in file:
            if "Step" in line:
                header = line.split()
                break

        # Define a function to process each chunk of data
        def process_chunk(chunk):
            chunk.columns = header
            return chunk.dropna()

        chunk_size = 10000  # Define a suitable chunk size
        chunks = pd.read_csv(file, sep="\s+", header=None, chunksize=chunk_size)

        for chunk in chunks:
            df_list.append(process_chunk(chunk))

    df = pd.concat(df_list, ignore_index=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    return df
