import numpy as np
from tqdm import tqdm
from numba import njit, prange
from typing import List, Dict, Optional, Union, Tuple


@njit(parallel=True, fastmath=True, boundscheck=False, nogil=True)
def calculate_sk_numba_optimized(k_vectors, pos_i, pos_j, norm):
    """
    Calculate static structure factor S(k) using Numba for optimized performance
    """
    n_k = k_vectors.shape[0]
    n_i = pos_i.shape[0]
    n_j = pos_j.shape[0]
    sk = np.zeros(n_k, dtype=np.float32)
    
    # Force memory continuity (key optimization!)
    pos_i = np.ascontiguousarray(pos_i)
    pos_j = np.ascontiguousarray(pos_j)
    k_vectors = np.ascontiguousarray(k_vectors)
    
    # Pre-expand k vector components (avoid accessing a 2D array each time)
    kx_all = k_vectors[:, 0]
    ky_all = k_vectors[:, 1]
    kz_all = k_vectors[:, 2]
    
    # Change the parallelization strategy to group by k
    for k in prange(n_k):
        kx = kx_all[k]
        ky = ky_all[k]
        kz = kz_all[k]
        
        sum_real_i = 0.0
        sum_imag_i = 0.0
        sum_real_j = 0.0
        sum_imag_j = 0.0
        
        for idx in range(max(n_i, n_j)):
            # Process particle i
            if idx < n_i:
                x_i = pos_i[idx, 0]
                y_i = pos_i[idx, 1]
                z_i = pos_i[idx, 2]
                dot_i = kx * x_i + ky * y_i + kz * z_i
                sum_real_i += np.cos(dot_i)
                sum_imag_i += np.sin(dot_i)
            
            # Process particle j
            if idx < n_j:
                x_j = pos_j[idx, 0]
                y_j = pos_j[idx, 1]
                z_j = pos_j[idx, 2]
                dot_j = kx * x_j + ky * y_j + kz * z_j
                sum_real_j += np.cos(dot_j)
                sum_imag_j += np.sin(dot_j)
        
        # Calculate the cross term
        sk[k] = (sum_real_i * sum_real_j + sum_imag_i * sum_imag_j) / norm
    
    return sk


class SsfCalc:
    """Calculate static structure factor from atomic configurations or RDF data"""
    
    def _apply_pbc(self, pos: np.ndarray, Lx: float, Ly: float, Lz: float) -> np.ndarray:
        """Apply periodic boundary conditions to positions"""
        pos_pbc = pos.copy()
        
        pos_pbc[:, 0] -= Lx * np.floor(pos_pbc[:, 0] / Lx)
        pos_pbc[:, 1] -= Ly * np.floor(pos_pbc[:, 1] / Ly)
        pos_pbc[:, 2] -= Lz * np.floor(pos_pbc[:, 2] / Lz)
        
        return pos_pbc
    
    def runSsf(
        self,
        coordinates: np.ndarray,
        bounds_matrix: np.ndarray,
        moltype: Union[List[int], np.ndarray],
        namemoltype: List[str],
        stable_steps: int,
        k_max: float = 15.0,
        output: Optional[Dict] = None,
        bin_precision: int = 9,
        ver: bool = True,
    ) -> Dict:
        """
        Calculate static structure factor S(k) for all species in the system

        Parameters
        ----------
        coordinates: A Three-dimensional array of floats with shape (Nframes, Natoms, 3)
        bounds_matrix: A two-dimensional array of floats with shape (3, 3)
        moltype : List or numpy array indicating the type of molecules
        namemoltype : List of molecule labels
        stable_steps : Number of frames to use after system relaxation
        k_max : Maximum k value for calculation, defaults to 15.0
        output : Optional dictionary to store results, defaults to None
        bin_precision : Determines how finely to bin k-magnitudes by specifying decimal rounding precision
        ver : Whether to show progress bar, defaults to True

        Returns
        -------
        Dict
            Dictionary containing S(k) results
        """
        # Initialize output dictionary
        if output is None:
            output = {}
        if 'S(k)_atomic' not in output:
            output['S(k)_atomic'] = {}

        comx, comy, comz = coordinates.transpose(2, 0, 1)
        Lx, Ly, Lz = bounds_matrix[0, 0], bounds_matrix[1, 1], bounds_matrix[2, 2]
        moltype = moltype - np.array(moltype).min() #start from 0! 

        L = max(Lx, Ly, Lz)
        dk = 2 * np.pi / L
        n_max = int(np.ceil(k_max / dk))
        # Generate all possible k vectors
        k_vectors = []
        for nx in range(n_max + 1):
            for ny in range(n_max + 1):
                for nz in range(n_max + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue  # Exclude k = 0 point
                    k_mag = dk * np.sqrt(nx**2 + ny**2 + nz**2)
                    if k_mag > k_max:
                        continue
                    k_vectors.append([nx * dk, ny * dk, nz * dk])
        
        if not k_vectors:
            raise ValueError("No valid k vectors generated. Check k_max and system size.")
        k_vectors = np.array(k_vectors, dtype=np.float32)
        k_magnitudes = np.linalg.norm(k_vectors, axis=1)
        # Use rounding for grouping
        rounded_magnitudes = np.round(k_magnitudes, bin_precision)
        unique_k, indices = np.unique(rounded_magnitudes, return_inverse=True)
        sorted_k_indices = np.argsort(unique_k)
        sorted_k_values = unique_k[sorted_k_indices]
        # Pre-generate indices for each k group
        group_indices = []
        for k_val in sorted_k_values:
            group_indices.append(np.where(rounded_magnitudes == k_val)[0])
        output['S(k)_atomic']['k'] = sorted_k_values

        unique_types = sorted(set(moltype))
        n_types = len(unique_types)
        total_pairs = n_types * (n_types + 1) // 2
        total_iterations = total_pairs * stable_steps

        with tqdm(total=total_iterations, desc="Calculating S(k)", disable=not ver) as pbar:
            for i in range(n_types):
                type_i = unique_types[i]
                for j in range(i, n_types):
                    type_j = unique_types[j]
                    
                    sk_sum = np.zeros_like(sorted_k_values, dtype=np.float32)
                    numsteps = len(comx)  
                    first_step = numsteps - stable_steps  
                
                    if first_step < 0:
                        print(f"Warning: stable_steps ({stable_steps}) is larger than total steps ({numsteps}). Using all steps.")
                        first_step = 0
                    
                    for step in range(first_step, numsteps):

                        pos = np.column_stack((comx[step], comy[step], comz[step]))
                        pos = self._apply_pbc(pos, Lx, Ly, Lz)
                        
                        mask_i = np.array(moltype) == type_i
                        mask_j = np.array(moltype) == type_j
                        pos_i = pos[mask_i]
                        pos_j = pos[mask_j]
                        
                        N_i = len(pos_i)
                        N_j = len(pos_j)
                        norm = np.sqrt(N_i * N_j) if i != j else N_i
                        if norm == 0:
                            sk_frame = np.zeros_like(sorted_k_values)
                        else:
                            sk_all = calculate_sk_numba_optimized(k_vectors.astype(np.float32),
                                                                    pos_i.astype(np.float32),
                                                                    pos_j.astype(np.float32),
                                                                    np.float32(norm))

                            # Average for each k group
                            sk_frame = np.array([np.mean(sk_all[indices]) for indices in group_indices], dtype=np.float32)
                        sk_sum += sk_frame
                        pbar.update(1)
                    sk_avg = sk_sum / stable_steps
                    pair_key = f"{namemoltype[type_i]}-{namemoltype[type_j]}"
                    output['S(k)_atomic'][pair_key] = sk_avg

        return output
    
    def rdf2ssf(
        self,
        k_points: np.ndarray,
        rdf_data: Dict[str, Dict],
        radii: np.ndarray,
        output: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate static structure factor (SSF) from radial distribution function (RDF).

        Parameters
        ----------
        k_points : np.ndarray
            Array of wave numbers (k)
        rdf_data : Dict[str, Dict]
            Dictionary containing RDF data for each pair:
            {
                "A-B": {
                    "g_r": array-like,  # RDF values
                    "rho_i": float,     # number density of component A
                    "rho_j": float      # number density of component B
                }
            }
        radii : np.ndarray
            Array of radial distances
        output : Optional[Dict]
            Optional dictionary to store results

        Returns
        -------
        Dict
            Dictionary containing S(k) results
        """
        if output is None:
            output = {}
        if 'S(k)_rdf_ft' not in output:
            output['S(k)_rdf_ft'] = {}
        
        output['S(k)_rdf_ft']['k'] = k_points
        
        # Pre-calculate broadcast shapes for efficiency
        k = k_points.reshape(-1, 1)
        r = radii.reshape(1, -1)
        dr = radii[1] - radii[0] if len(radii) > 1 else 0

        for pair_key, params in rdf_data.items():
            g_r = np.asarray(params['g_r'])
            rho_i = params['rho_i']
            rho_j = params['rho_j']
            
            # Check if it's self-correlation (A-A) or cross-correlation (A-B)
            species = pair_key.split('-')
            delta = 1.0 if species[0] == species[1] else 0.0

            # Calculate the integrand: (g_ij(r) - 1) * r * sin(k * r)
            integrand = (g_r - 1) * r * np.sin(k * r)
            integral = dr * np.sum(integrand, axis=1)

            sqrt_rho = np.sqrt(rho_i * rho_j)
            
            # Calculate S(k) = δ_ij + (4π√(ρ_iρ_j)/k) * ∫[(g_ij(r)-1) r sin(kr) dr]
            with np.errstate(divide='ignore', invalid='ignore'):
                s_k = delta + (4 * np.pi * sqrt_rho / k.squeeze()) * integral
                # Handle k = 0 case
                s_k = np.where(k.squeeze() != 0, s_k, delta)

            output['S(k)_rdf_ft'][pair_key] = s_k

        return output