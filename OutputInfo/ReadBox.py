import numpy as np

def read_XDATCAR_frame(filename, interval=1):
    frames = []
    with open(filename, 'r') as XDATCAR_file:
        system_info = XDATCAR_file.readline()
        scale_factor = int(float(XDATCAR_file.readline().strip()))
        cell_vectors = []
        for _ in range(3):
            cell_vectors.append([scale_factor * float(x) for x in XDATCAR_file.readline().split()])
        cell_vectors = np.array(cell_vectors)
        atoms_types = XDATCAR_file.readline().split()[0]
        num_atom = [int(x) for x in XDATCAR_file.readline().split()]
        
        while True:
            try:
                info_line = XDATCAR_file.readline().split()
                if len(info_line) > 1:
                    coord_type, step = info_line[0], int(info_line[2])
                else:
                    coord_type, step = info_line[0], 0

                if step % interval != 0:
                    for _ in range(np.array(num_atom).sum()):
                        XDATCAR_file.readline()
                    continue

                coordinate = []
                for _ in range(np.array(num_atom).sum()):
                    atom_info = np.array([float(x) for x in XDATCAR_file.readline().split()])
                    if coord_type == 'Direct':
                        atom_info %= 1  
                        atom_info = atom_info @ cell_vectors
                    else:
                        fractional = np.linalg.solve(cell_vectors.T, atom_info) % 1
                        atom_info = fractional @ cell_vectors
                    coordinate.append(atom_info)
                
                frames.append((step, np.array(coordinate)))
            
            except:
                print(f"reading frame step {step}")
                break

    if len(frames) == 1:
        return frames[0][1], cell_vectors, num_atom
    else:
        return frames, cell_vectors, num_atom

def read_xyz_frame( file):
    """
    Read from xyz file.
    
    Parameters:
        file : XYZ file

    Returns:
        atom_type : list
        np.array(coordinate) : num_atom*3
        np.array(cell_length) : array,1*3
    """
    num_atom = int(file.readline())
    try:
        cell_length = [float(i) for i in file.readline().split()]
    except:
        cell_length = None
    
    atom_type = []
    coordinate = []

    for _ in range(num_atom):
        atom_info = file.readline().split()
        atom_type.append(atom_info[0])
        coordinate.append([float(coord) for coord in atom_info[1:]])
        
    return atom_type,coordinate,cell_length
    """
    --------------------------usage example-----------------------------
    frames = []
    with open('name.xyz', 'r') as xyz_file:
        while True:
            try:
                _, coordinate,cell_length= read_xyz_frame(xyz_file)
                frames.append((coordinate,cell_length))
            except ValueError:
                break
    --------------------------------------------------------------------
    """

def read_lammpstrj_frame( file, interval=1):
    """
    Read lammpstrj file with a specified interval.
    
    Parameters:
        file : file object
            Opened lammpstrj file
        interval : int, optional
            Interval to control how often frames are read. 
            Default is 1, meaning read every frame.
    
    Returns:
        frames : list of tuples
            Each tuple contains (step, coordinate_sorted, bounds_matrix)
            coordinate is absolute.
        atom_type_global : list of int
            sorted global atom type
    """
    frames = []
    atom_type_global = None  # Initialize outside loop

    while True:
        line = file.readline()
        if not line:
            break  # End of file reached
        
        if "ITEM: TIMESTEP" in line:
            step = int(file.readline())
            file.readline()  # Skip ITEM: NUMBER OF ATOMS line

            if step % interval != 0:
                nn = int(file.readline())
                for _ in range(5 + nn):  # Skip the rest of this timestep
                    file.readline()
                continue
            
            num_atom = int(file.readline())
            box_bounds_line = file.readline()
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

            info_line = file.readline().split()

            if any(keyword in info_line for keyword in ['xs', 'ys', 'zs']):
                coordtype='relative'
            else:
                coordtype='absolute'

            atom_data = []
            for _ in range(num_atom):
                atom_info = file.readline().split()
                atom_id = int(atom_info[0])
                atom_type = int(atom_info[1])
                x, y, z = [float(coord) for coord in atom_info[2:]]

                if coordtype=='relative':
                    x = x*bounds_matrix[0, 0] + min_bound_x
                    y = y*bounds_matrix[1, 1] + min_bound_y
                    z = z*bounds_matrix[2, 2] + min_bound_z

                wrap_x = (x - min_bound_x) % bounds_matrix[0, 0]
                wrap_y = (y - min_bound_y) % bounds_matrix[1, 1]
                wrap_z = (z - min_bound_z) % bounds_matrix[2, 2]

                atom_data.append((atom_id, atom_type, wrap_x, wrap_y, wrap_z))

            # Sorting by atom index
            atom_data_sorted = sorted(atom_data, key=lambda x: x[0])
            coordinate_sorted = np.array([[data[2], data[3], data[4]] for data in atom_data_sorted])
            
            if atom_type_global is None:
                atom_type_global = [data[1] for data in atom_data_sorted]  # Extract atom types once

            frames.append((step, coordinate_sorted, bounds_matrix))

    if len(frames) == 1:
        return frames[0], atom_type_global
    else:
        return frames, atom_type_global


    """
    --------------------------usage example (if multiple lammpstrj files)-----------------------------
    frames = []
    for i in range(0,100000,10):
        traj = tmp_dir+'/traj/'+str(i)+'.lammpstrj'
        with open(traj, 'r') as lammpstrj_file:
            frames,atom_type = read_lammpstrj_frame(lammpstrj_file)
            step,coordinate,bounds_matrix = frames
            cell_vectors = np.diag(bounds_matrix)
            frames.append((coordinate,cell_vectors))
    --------------------------------------------------------------------------------------------------

    --------------------------usage example (if single lammpstrj file)--------------------------------
    interval = 5
    with open(trjfilename, 'r') as lammpstrj_file:
        frames,atom_type = read_lammpstrj_frame(lammpstrj_file,interval)
    steps, coordinates_list, bounds_matrices = zip(*frames)
    --------------------------------------------------------------------------------------------------

    """