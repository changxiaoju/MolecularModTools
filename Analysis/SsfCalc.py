import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from numba import njit, prange
from typing import List, Dict, Optional, Union, Tuple

@njit(parallel=True, fastmath=True)
def calculate_sk_numba(k_points, k_directions, pos_i, pos_j, norm):
    n_k = len(k_points)
    n_dir = len(k_directions)
    sk = np.zeros(n_k, dtype=np.float32)
    
    for k in prange(n_k):
        k_val = k_points[k]
        sum_real = 0.0
        sum_imag = 0.0
        
        for d in range(n_dir):
            kx, ky, kz = k_val * k_directions[d]
            
            # 预计算三角函数参数
            dot_i = pos_i[:, 0] * kx + pos_i[:, 1] * ky + pos_i[:, 2] * kz
            dot_j = pos_j[:, 0] * kx + pos_j[:, 1] * ky + pos_j[:, 2] * kz
            
            # 使用欧拉公式展开复数运算
            sum_i_real = np.sum(np.cos(dot_i))
            sum_i_imag = np.sum(np.sin(dot_i))
            sum_j_real = np.sum(np.cos(dot_j))
            sum_j_imag = np.sum(np.sin(dot_j))
            
            # 计算点积的实部
            real_part = (sum_i_real * sum_j_real + sum_i_imag * sum_j_imag)
            sum_real += real_part
        
        sk[k] = sum_real / (n_dir * norm)
    
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
        n_k_directions: int = 50,
        output: Optional[Dict] = None,
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
        n_k_directions : Number of k directions for spherical averaging, defaults to 50
        output : Optional dictionary to store results, defaults to None
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

        # Initialize k points
        k_min = 2 * np.pi / (max(Lx, Ly, Lz))
        k_points = np.arange(k_min, k_max + k_min, k_min)
        output['S(k)_atomic']['k'] = k_points

        # Pre-generate k directions
        k_directions = self._generate_k_directions(n_k_directions).astype(np.float32)

        unique_types = sorted(set(moltype))
        n_types = len(unique_types)

        with tqdm(total=n_types*(n_types+1)//2*stable_steps, desc="Calculating S(k)", disable=not ver) as pbar:
            for i in range(n_types):
                type_i = unique_types[i]
                for j in range(i, n_types):
                    type_j = unique_types[j]
                    
                    sk_sum = np.zeros_like(k_points)
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
                        
                        sk_frame = calculate_sk_numba(k_points.astype(np.float32), 
                                                        k_directions,
                                                        pos_i, 
                                                        pos_j,
                                                        np.float32(norm))
                        
                        sk_sum += sk_frame
                        pbar.update(1)
                    
                    sk_avg = sk_sum / stable_steps
                    
                    pair_key = f"{namemoltype[type_i]}-{namemoltype[type_j]}"
                    output['S(k)_atomic'][pair_key] = sk_avg

        return output
    
    @staticmethod
    def _generate_k_directions(n_directions: int) -> np.ndarray:
        """Generate uniformly distributed unit vectors for k directions"""
        golden_ratio = (1 + np.sqrt(5)) / 2
        i = np.arange(n_directions)
        phi = 2 * np.pi * i / golden_ratio
        cos_theta = 1 - 2 * (i + 0.5) / n_directions
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        kx = sin_theta * np.cos(phi)
        ky = sin_theta * np.sin(phi)
        kz = cos_theta
        
        return np.column_stack((kx, ky, kz))
    

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

