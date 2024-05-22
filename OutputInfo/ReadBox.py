import numpy as np

def read_XDATCAR_frame(file,num_atoms,cell_vectors):
    """
    只从XDATCAR文件中读取原子坐标。
    
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

    return np.array(coordinates),cell_length
    """
    --------------------------usage example-----------------------------
    frames,Cell_length = [],[]
    with open('XDATCAR', 'r') as XDATCAR_file:
        system_info = XDATCAR_file.readline() #一般啥也不是
        scale_factor = int(XDATCAR_file.readline().strip())
        cell_vectors = []
        for _ in range(0,3):
            cell_vectors.append([scale_factor*float(x) for x in XDATCAR_file.readline().split()])
        atoms_types = XDATCAR_file.readline().split()
        num_atoms = [int(x) for x in XDATCAR_file.readline().split()]
        Cell_length.append(cell_vectors[0][0])
        while True:
            try:
                coordinates,cell_length = read_XDATCAR_frame(XDATCAR_file,num_atoms,cell_vectors)
                frames.append((coordinates,cell_length))
            except:
                break
    --------------------------------------------------------------------
    """
def read_xyz_frame(file):
    """
    只从xyz文件中读取原子坐标。
    
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
def read_lammpstrj_frame(file):
    """
    只从lammpstrj文件中读取原子坐标。
    
    Parameters:
        file : lammpstrj file

    Returns:
        step : int
        np.array(cell_length) : array,1*3
        np.array(atom_index) : ...
        atom_types : list
        np.array(coordinates) : num_atoms*3
        bounds_matrix : array
            min_bound_x,max_bound_x,skew_xy
            min_bound_y,max_bound_y,skew_xz
            min_bound_z,max_bound_z,skew_yz
    """
    file.readline() #ITEM: TIMESTEP
    step = int(file.readline())
    
    file.readline() #ITEM: NUMBER OF ATOMS
    num_atoms = int(file.readline())
    
    file.readline() #ITEM: BOX BOUNDS xy xz yz pp pp pp
    min_bound_x,max_bound_x,skew_xy = [float(i) for i in file.readline().split()]
    min_bound_y,max_bound_y,skew_xz = [float(i) for i in file.readline().split()]
    min_bound_z,max_bound_z,skew_yz = [float(i) for i in file.readline().split()]
    bounds_list = [min_bound_x, max_bound_x, skew_xy, min_bound_y, max_bound_y, skew_xz, min_bound_z, max_bound_z, skew_yz]
    bounds_array = np.array(bounds_list)
    bounds_matrix = bounds_array.reshape((3, 3))
    
    file.readline() #ITEM: ATOMS id type x y z
    atom_types,atom_index,coordinates = [],[],[]

    for _ in range(num_atoms):
        atom_info = file.readline().split()
        atom_index.append(int(atom_info[0]))
        atom_types.append(atom_info[1])
        x,y,z = [float(coord) for coord in atom_info[2:]]
        #The input data shall be wrapped into [0,l) for KDtree
        wrap_x = (x - min_bound_x) % (max_bound_x - min_bound_x)
        wrap_y = (y - min_bound_y) % (max_bound_y - min_bound_y)
        wrap_z = (z - min_bound_z) % (max_bound_z - min_bound_z)
        coordinates.append([wrap_x,wrap_y,wrap_z])
        
    return step,np.array(atom_index),atom_types,np.array(coordinates),bounds_matrix

def read_lammpstrj_NVT_frame(file):
    """
    只从lammpstrj文件中读取原子坐标。
    NVT系综盒子
    
    Parameters:
        file : lammpstrj file

    Returns:
        step : int
        np.array(cell_length) : array,1*3
        np.array(atom_index) : ...
        atom_types : list
        np.array(coordinates) : num_atoms*3
        bounds_matrix : array
            min_bound_x,max_bound_x
            min_bound_y,max_bound_y
            min_bound_z,max_bound_z
    """
    file.readline() #ITEM: TIMESTEP
    step = int(file.readline())
    
    file.readline() #ITEM: NUMBER OF ATOMS
    num_atoms = int(file.readline())
    
    file.readline() #ITEM: BOX BOUNDS pp pp pp
    min_bound_x,max_bound_x = [float(i) for i in file.readline().split()]
    min_bound_y,max_bound_y = [float(i) for i in file.readline().split()]
    min_bound_z,max_bound_z = [float(i) for i in file.readline().split()]
    bounds_list = [min_bound_x, max_bound_x,0,min_bound_y, max_bound_y, 0,min_bound_z, max_bound_z,0]
    bounds_array = np.array(bounds_list)
    bounds_matrix = bounds_array.reshape((3, 3))
    
    file.readline() #ITEM: ATOMS id type x y z
    atom_types,atom_index,coordinates = [],[],[]

    for _ in range(num_atoms):
        atom_info = file.readline().split()
        atom_index.append(int(atom_info[0]))
        atom_types.append(atom_info[1])
        x,y,z = [float(coord) for coord in atom_info[2:]]
        #The input data shall be wrapped into [0,l) for KDtree
        wrap_x = (x - min_bound_x) % (max_bound_x - min_bound_x)
        wrap_y = (y - min_bound_y) % (max_bound_y - min_bound_y)
        wrap_z = (z - min_bound_z) % (max_bound_z - min_bound_z)
        coordinates.append([wrap_x,wrap_y,wrap_z])
        
    return step,np.array(atom_index),atom_types,np.array(coordinates),bounds_matrix
    """
    --------------------------usage example-----------------------------
    frames = []
    for i in range(0,100000,10):
        traj = tmp_dir+'/traj/'+str(i)+'.lammpstrj'
        with open(traj, 'r') as lammpstrj_file:
            step,atom_index,atom_types,coordinates,bounds_matrix = read_lammpstrj_frame(lammpstrj_file)
            cell_vectors = bounds_matrix[:,1]-bounds_matrix[:,0]
            frames.append((coordinates,cell_vectors))
    --------------------------------------------------------------------
    """