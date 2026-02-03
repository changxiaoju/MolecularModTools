import numpy as np
from typing import Dict, Tuple, Optional, List

class XrdCalc:
    """
    XRD Pattern Calculator.
    
    Framework: Based on your original XrdCalc module (Multi-element, Cromer-Mann coefficients).
    Physics:   Based on your 'XrdFinal' logic (Geometric Volume Correction canceling LP denominator).
    """
    def __init__(self):
        # Cromer-Mann coefficients (a1, b1, a2, b2, a3, b3, a4, b4, c)
        # Source: International Tables for Crystallography, Vol. C
        self.atomic_form_factors = {
            'H':  [0.489918, 20.6593, 0.262003, 7.74039, 0.196767, 49.5519, 0.049879, 2.20159, 0.001305],
            'He': [0.8734, 9.1037, 0.6309, 3.3568, 0.3112, 22.9276, 0.1780, 0.9821, 0.0064],
            'Li': [1.1282, 3.9546, 0.7508, 1.0524, 0.6175, 85.3905, 0.4653, 168.261, 0.0377],
            'Be': [1.5919, 2.4420, 1.1278, 0.3756, 0.5391, 74.4980, 0.7029, 186.255, 0.0385],
            'B':  [2.0545, 23.2185, 1.3326, 1.0210, 1.0979, 60.3498, 0.7068, 0.1403, -0.1932],
            'C':  [2.3100, 20.8439, 1.0200, 10.2075, 1.5886, 0.5687, 0.8650, 51.6512, 0.2156],
            'N':  [12.2126, 0.0057, 3.1322, 9.8933, 2.0125, 28.9975, 1.1663, 0.5826, -11.529],
            'O':  [3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.8670, 32.9089, 0.2508],
            'F':  [3.5392, 10.2825, 2.6412, 4.2944, 1.5170, 0.2615, 1.0243, 26.1476, 0.2776],
            'Ne': [3.9553, 8.4042, 3.1125, 3.4262, 1.4546, 0.2306, 1.1251, 21.7184, 0.3515],
            'Na': [4.7626, 3.2850, 3.1736, 8.8422, 1.2674, 0.3136, 1.1128, 129.424, 0.6760],
            'Mg': [5.4204, 2.8275, 2.1735, 79.2611, 1.2269, 0.3808, 2.3073, 7.1937, 0.8584],
            'Al': [6.4202, 3.0387, 1.9002, 0.7426, 1.5936, 31.5472, 1.9646, 85.0886, 1.1151],
            'Si': [6.2915, 2.4386, 3.0353, 32.333, 1.9891, 0.6785, 1.5410, 81.6981, 1.1407],
            'P':  [6.4345, 1.9067, 4.1791, 27.1570, 1.7800, 0.5260, 1.4908, 68.1645, 1.1149],
            'S':  [6.9053, 1.4679, 5.2034, 22.2151, 1.4379, 0.2536, 1.5863, 56.1720, 0.8669],
            'Cl': [11.4604, 0.0104, 7.1964, 1.1662, 6.3184, 18.5576, 1.6486, 47.7784, -9.625],
            'Ar': [7.4845, 16.7597, 6.7723, 1.0631, 0.6539, 142.1121, 1.6442, 39.3887, 1.4445],
            'K':  [8.2186, 12.7949, 7.4398, 0.7748, 1.0519, 213.187, 0.8659, 41.6841, 1.4228],
            'Ca': [8.6266, 10.4421, 7.3873, 0.6599, 1.5899, 85.7484, 1.0211, 178.437, 1.3751],
            'Fe': [11.045, 4.6538, 7.1234, 0.3043, 4.2541, 19.584, 1.6437, 83.8463, 1.9598],
        }

    def get_atomic_form_factor(self, element: str, k_values: np.ndarray) -> np.ndarray:
        """Calculate atomic form factor f(k)"""
        if element not in self.atomic_form_factors:
            raise ValueError(f"Element {element} not found in database.")
        
        coeffs = self.atomic_form_factors[element]
        # s = sin(theta)/lambda = k / 4pi
        s_sq = (k_values / (4 * np.pi)) ** 2
        
        f_k = np.full_like(k_values, coeffs[8]) # term c
        for i in range(4):
            f_k += coeffs[i*2] * np.exp(-coeffs[i*2+1] * s_sq)
            
        return f_k

    def runXrd(self,
            composition: Dict[str, int],
            output: Dict,
            wavelength: float = 0.7093) -> Dict:
        """
        Calculate XRD pattern applying the Geometric Volume Correction (Jacobian).

        Logic adopted from XrdFinal:
        I_final = I_scattering * LP * Jacobian
        Since LP ~ (1+cos^2)/(sin^2*cos) and Jacobian ~ sin^2*cos
        They cancel out to leave only: (1 + cos^2(2theta))

        Parameters
        ----------
        composition : Dict[str, int], dictionary of element counts (e.g., {'H': 100, 'O': 50})
        output : Output dictionary from previous calculations, must contain 'S(k)_atomic'
        wavelength : float, X-ray wavelength in Angstrom

        Returns
        -------
        Dict with '2theta' and 'I' arrays, also writes to output['XRD'].
        """
        
        if 'S(k)_atomic' not in output:
            raise ValueError("Output dictionary must contain 'S(k)_atomic' data.")

        k_values = np.array(output['S(k)_atomic']['k'])
        
        # 1. Convert k to 2Theta
        # k = 4*pi*sin(theta) / lambda
        sin_theta = k_values * wavelength / (4 * np.pi)
        
        # Filter valid sin_theta values (<= 1.0)
        valid_mask = sin_theta <= 1.0
        k_valid = k_values[valid_mask]
        sin_theta = sin_theta[valid_mask]
        
        theta_rad = np.arcsin(sin_theta)
        two_theta_deg = 2 * np.degrees(theta_rad)
        
        # 2. Calculate Weighted Atomic Scattering Intensity (I_density)
        # This is the "Skeleton" logic: handling multi-element weighting

        total_atoms = sum(composition.values())
        I_scattering = np.zeros_like(k_valid)

        # Pre-calculate f(k) and fractions
        f_factors = {}
        fractions = {}
        for elem, count in composition.items():
            f_factors[elem] = self.get_atomic_form_factor(elem, k_valid)
            fractions[elem] = count / total_atoms

        # Weighted Sum of Partial S(k)
        # Formula: Sum( sqrt(xi*xj) * fi * fj * S_ij(k) )
        for pair_key, sk_vals in output['S(k)_atomic'].items():
            if pair_key == 'k': continue
            
            # Align data size with valid mask
            sk_vals = np.array(sk_vals)[valid_mask]
            
            elem_a, elem_b = pair_key.split('-')
            
            # Calculate weight
            term_weight = np.sqrt(fractions[elem_a] * fractions[elem_b])
            fa = f_factors[elem_a]
            fb = f_factors[elem_b]
            
            # Multiply by 2 for cross-terms (A-B), 1 for self-terms (A-A)
            # if the S(k) input treats A-B and B-A as one curve
            multiplicity = 2.0 if elem_a != elem_b else 1.0
            
            I_scattering += multiplicity * term_weight * fa * fb * sk_vals

        # 3. Apply the "Magic" Correction (Snippet 2 Logic)
        # LP Factor = (1 + cos^2(2theta)) / (sin^2(theta) * cos(theta))
        # Jacobian  = sin^2(theta) * cos(theta)   <-- Geometric volume correction
        # Final Correction = LP * Jacobian = 1 + cos^2(2theta)
        
        combined_correction = 1 + np.cos(2 * theta_rad) ** 2
        
        I_final = I_scattering * combined_correction

        # Handle index 0 (theta=0) explicitly if needed, though cos(0)=1 is safe

        output['XRD'] = {
            "2theta": two_theta_deg.tolist(),
            "I": I_final.tolist(),
            "wavelength": wavelength
        }


        return output