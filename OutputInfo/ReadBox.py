import numpy as np

def read_XDATCAR_frame(file,num_atoms,cell_vectors):
    """
    Read from XDATCAR(or POSCAR).
    
    Parameters:
        file : XDATCAR
        num_atoms : int
        cell_vectors : list
    Returns:
        np.array(coordinates) : num_atoms*3
        cell_length : array,1*3
    """
    
    coord_type = file.readline().split()[0]
    coordinates = []
    for _ in range(np.array(num_atoms).sum()):
        
        cell_length = np.linalg.norm(np.array(cell_vectors), axis=1)
        atom_info = file.readline().split()
        if coord_type=='Direct':
            coordinates.append([float(coord)%1 for coord in atom_info]*cell_length)
        else: # To implement: wrap Cartesian coordinates
            coordinates.append([float(coord) for coord in atom_info])

    return cell_length,np.array(coordinates)
    """
    --------------------------usage example-----------------------------
    frames,Cell_length = [],[]
    with open('XDATCAR', 'r') as XDATCAR_file:
        system_info = XDATCAR_file.readline() 
        scale_factor = int(float(XDATCAR_file.readline().strip())) 
        cell_vectors = []
        for _ in range(0,3):
            cell_vectors.append([scale_factor*float(x) for x in XDATCAR_file.readline().split()])
        atoms_types = XDATCAR_file.readline().split()
        num_atoms = [int(x) for x in XDATCAR_file.readline().split()]
        Cell_length.append(cell_vectors[0][0])
        while True:
            try:
                cell_length,coordinates = read_XDATCAR_frame(XDATCAR_file,num_atoms,cell_vectors)
                frames.append((coordinates,cell_length))
            except:
                break
    --------------------------------------------------------------------
    """
def read_xyz_frame(file):
    """
    Read from xyz file.
    
    Parameters:
        file : XYZ file

    Returns:
        atom_types : list
        np.array(coordinates) : num_atoms*3
        np.array(cell_length) : array,1*3
    """
    num_atoms = int(file.readline())
    try:
        cell_length = [float(i) for i in file.readline().split()]
    except:
        cell_length = None
    
    atom_types = []
    coordinates = []

    for _ in range(num_atoms):
        atom_info = file.readline().split()
        atom_types.append(atom_info[0])
        coordinates.append([float(coord) for coord in atom_info[1:]])
        
    return atom_types,coordinates,cell_length
    """
    --------------------------usage example-----------------------------
    frames = []
    with open('name.xyz', 'r') as xyz_file:
        while True:
            try:
                _, coordinates,cell_length= read_xyz_frame(xyz_file)
                frames.append((coordinates,cell_length))
            except ValueError:
                break
    --------------------------------------------------------------------
    """

def read_lammpstrj_frame(file, interval=1):
    """
    Read lammpstrj file with a specified interval.
    
    Parameters:
        file : file object
            Opened lammpstrj file
        interval : int, optional
            Interval to control how often frames are read. 
            Default is 1, meaning read every frame.
    
    Returns:
        all_frames : list of tuples
            Each tuple contains (step, atom_index, atom_types, coordinates, bounds_matrix)
    """
    all_frames = []
    
    while True:
        line = file.readline()
        if not line:
            break  # End of file reached
        
        if "ITEM: TIMESTEP" in line:
            step = int(file.readline())
            file.readline()  # ITEM: NUMBER OF ATOMS

            if step % interval != 0:
                # Skip this timestep and all related data
                nn = int(file.readline())
                for _ in range(5 + nn):  # ITEM: BOX BOUNDS xy xz yz pp pp pp (4 rows); ITEM: ATOMS id type x y z
                    file.readline()
                continue
            
            
            num_atoms = int(file.readline())
            
            box_bounds_line = file.readline()  # ITEM: BOX BOUNDS [xy xz yz] pp pp pp
            is_triclinic = 'xy' in box_bounds_line
            
            if is_triclinic:
                min_bound_x, max_bound_x, skew_xy = [float(i) for i in file.readline().split()]
                min_bound_y, max_bound_y, skew_xz = [float(i) for i in file.readline().split()]
                min_bound_z, max_bound_z, skew_yz = [float(i) for i in file.readline().split()]
                
                bounds_matrix = np.array([
                    [max_bound_x - min_bound_x, skew_xy, skew_xz],
                    [0, max_bound_y - min_bound_y, skew_yz],
                    [0, 0, max_bound_z - min_bound_z]
                ])
            else:
                min_bound_x, max_bound_x = [float(i) for i in file.readline().split()]
                min_bound_y, max_bound_y = [float(i) for i in file.readline().split()]
                min_bound_z, max_bound_z = [float(i) for i in file.readline().split()]
                
                bounds_matrix = np.array([
                    [max_bound_x - min_bound_x, 0, 0],
                    [0, max_bound_y - min_bound_y, 0],
                    [0, 0, max_bound_z - min_bound_z]
                ])
            
            file.readline()  # ITEM: ATOMS id type x y z
            atom_types, atom_index, coordinates = [], [], []

            for _ in range(num_atoms):
                atom_info = file.readline().split()
                atom_index.append(int(atom_info[0]))
                atom_types.append(atom_info[1])
                x, y, z = [float(coord) for coord in atom_info[2:]]
                
                wrap_x = (x - min_bound_x) % (bounds_matrix[0, 0])
                wrap_y = (y - min_bound_y) % (bounds_matrix[1, 1])
                wrap_z = (z - min_bound_z) % (bounds_matrix[2, 2])
                coordinates.append([wrap_x, wrap_y, wrap_z])
            
            all_frames.append((step, np.array(atom_index), atom_types, np.array(coordinates), bounds_matrix))
    
    return all_frames


    """
    --------------------------usage example (if multiple lammpstrj files)-----------------------------
    frames = []
    for i in range(0,100000,10):
        traj = tmp_dir+'/traj/'+str(i)+'.lammpstrj'
        with open(traj, 'r') as lammpstrj_file:
            step,atom_index,atom_types,coordinates,bounds_matrix = read_lammpstrj_frame(lammpstrj_file)[0] # defult interval = 1
            cell_vectors = np.diag(bounds_matrix)
            frames.append((coordinates,cell_vectors))
    --------------------------------------------------------------------
    """