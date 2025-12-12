import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Tuple, Any

class MsdCalc:

    def runMsd(
        self,
        coordinates: np.ndarray,
        bounds_matrix: np.ndarray,
        moltype: Union[List[int], np.ndarray],
        namemoltype: List[str],
        dt: float,
        skip: int,
        per_atom: bool = False,   
        batch_size: int = 10000,
        num_init: Optional[int] = None,
        output: Optional[Dict] = None,
        ver: bool = True 
    ) -> Dict:
        """
        Calculate mean squared displacement (MSD) for all species in the system.
        If per_atom=False (default): calculates averaged MSD by molecule type.
        If per_atom=True: calculates MSD for each individual atom (large output, beware of memory).

        Parameters:
            coordinates: A three-dimensional array of floats with shape (Nframes, Natoms, 3)
            bounds_matrix: A two-dimensional array of floats with shape (3, 3)
            moltype: List or numpy array indicating the type of molecules
            namemoltype: List of molecule labels
            dt: Time step between frames
            skip: Number of initial frames to skip
            num_init: Number of initial time origins to use for averaging
            output: Optional dictionary to store results
            ver: Boolean to enable/disable progress bar
            per_atom: Whether to calculate per-atom MSD instead of averaged MSD
            batch_size: Batch size for processing atoms when per_atom=True

        Returns:
            Dict: Updated output dictionary containing MSD results
        """
        if output is None:
            output = {}

        Lx, Ly, Lz = bounds_matrix[0, 0], bounds_matrix[1, 1], bounds_matrix[2, 2]
        box = np.array([Lx, Ly, Lz])
        
        if ver: print("Unwrapping coordinates...")
        unwrapped_coords = self.unwrap_vectorized(coordinates, box)
        
        comx = unwrapped_coords[:, :, 0]
        comy = unwrapped_coords[:, :, 1]
        comz = unwrapped_coords[:, :, 2]

        num_timesteps, num_atoms = comx.shape
        
        moltype = np.array(moltype)
        moltype = moltype - moltype.min()

        if num_init is None:
            num_init = int(np.floor((num_timesteps - skip) / 2))
        else:
            num_init = int(num_init)
            
        len_MSD = num_timesteps - skip - num_init
        
        Time = np.arange(len_MSD) * dt


        if "MSD" not in output:
            output["MSD"] = {"Units": "Angstroms^2, ps", "Time": Time.tolist()}

        unique_types = np.unique(moltype)
        
        for m_idx in unique_types:
            type_name = namemoltype[m_idx]
            atom_indices = np.where(moltype == m_idx)[0]
            n_atoms_this_type = len(atom_indices)
            
            if n_atoms_this_type == 0:
                continue

            if ver: print(f"Calculating Average MSD for type: {type_name} ({n_atoms_this_type} atoms)")

            sub_x = comx[:, atom_indices]
            sub_y = comy[:, atom_indices]
            sub_z = comz[:, atom_indices]

            msd_accum = np.zeros((4, len_MSD))

            with tqdm(total=num_init, desc=f"MSD {type_name}", disable=not ver) as pbar:
                for i in range(skip, num_init + skip):
                    dx = sub_x[i:i+len_MSD, :] - sub_x[i, :]
                    dy = sub_y[i:i+len_MSD, :] - sub_y[i, :]
                    dz = sub_z[i:i+len_MSD, :] - sub_z[i, :]

                    dx2 = dx**2
                    dy2 = dy**2
                    dz2 = dz**2
                    d_tot2 = dx2 + dy2 + dz2

                    msd_accum[0] += np.sum(dx2, axis=1)
                    msd_accum[1] += np.sum(dy2, axis=1)
                    msd_accum[2] += np.sum(dz2, axis=1)
                    msd_accum[3] += np.sum(d_tot2, axis=1)
                    
                    pbar.update(1)

            msd_accum /= (num_init * n_atoms_this_type)

            output["MSD"][type_name] = {
                "x": msd_accum[0].tolist(),
                "y": msd_accum[1].tolist(),
                "z": msd_accum[2].tolist(),
                "total": msd_accum[3].tolist()
            }


        if per_atom:
            if ver: print(f"Calculating Per-Atom MSD (Batch size: {batch_size})...")
            
            all_atoms_msd = [] 
            
            for b_start in range(0, num_atoms, batch_size):
                b_end = min(b_start + batch_size, num_atoms)
                n_batch = b_end - b_start
                
                batch_msd_accum = np.zeros((n_batch, len_MSD))
                
                bx = comx[:, b_start:b_end]
                by = comy[:, b_start:b_end]
                bz = comz[:, b_start:b_end]

                with tqdm(total=num_init, desc=f"Batch {b_start}-{b_end}", disable=not ver) as pbar:
                    for i in range(skip, num_init + skip):
                        diff_x = bx[i:i+len_MSD, :] - bx[i, :]
                        diff_y = by[i:i+len_MSD, :] - by[i, :]
                        diff_z = bz[i:i+len_MSD, :] - bz[i, :]
                        
                        r2 = diff_x**2 + diff_y**2 + diff_z**2
                        
                        batch_msd_accum += r2.T
                        
                        pbar.update(1)
                
                batch_msd_accum /= num_init
                all_atoms_msd.append(batch_msd_accum)

            final_msd = np.vstack(all_atoms_msd)
            
            output["MSD_i"] = {
                "Units": "Angstroms^2",
                "Time": Time.tolist(),
                "Data": final_msd.tolist()
            }

        return output

    def unwrap_vectorized(self, coords: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Vectorized unwrapping of coordinates using periodic boundary conditions.
        Much faster than looping through atoms individually.

        Parameters:
            coords: Three-dimensional array with shape (Nframes, Natoms, 3)
            box: One-dimensional array with box dimensions [Lx, Ly, Lz]

        Returns:
            np.ndarray: Unwrapped coordinates with shape (Nframes, Natoms, 3)
        """
        delta = np.diff(coords, axis=0)
        
        delta -= np.round(delta / box) * box
        
        unwrapped_trajectory = np.cumsum(delta, axis=0)
        
        unwrapped = np.vstack((
            coords[0][np.newaxis, :, :], 
            coords[0] + unwrapped_trajectory
        ))
        
        return unwrapped