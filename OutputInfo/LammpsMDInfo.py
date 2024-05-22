import pandas as pd
import re
from io import StringIO

def read_data(log_file):
    """
    
    Parameters
    ----------
    log_file: string
    Name of lammps MD log file
        
    Returns
    -------
    df: pd.DataFrame
    The dataframe with header like "Step Temp PotEng..."
    
    """
    with open(log_file, 'r') as file:
        log_content = file.read()

    table_pattern = re.compile(r'Step(.*?)\n\n', re.DOTALL)
    table_matches = table_pattern.findall(log_content)

    if table_matches:
        table_content = table_matches[0].strip()
        df = pd.read_csv(StringIO(table_content), delim_whitespace=True)
        df = df.drop('Loop', axis=0)
        return df