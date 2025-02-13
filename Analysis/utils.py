import scipy.signal as scp_signal
import numpy as np
from typing import Union, Sequence


def correlationfunction(
    At: Union[np.ndarray, Sequence[float]], 
    Bt: Union[np.ndarray, Sequence[float]]
) -> np.ndarray:
    """
    Calculate the correlation function between A(t) and B(t) using scipy.signal.correlate

    Parameters:
        At: First observable to correlate
        Bt: Second observable to correlate

    Returns:
        np.ndarray: Correlation function C_AB(τ)

    Examples:
        >>> t = np.linspace(0.0, 6.0 * np.pi, 3000)
        >>> w0 = 0.5
        >>> At = np.cos(w0 * t)
        >>> Bt = np.sin(w0 * t)
        >>> corr_t = correlationfunction(At, Bt)
    """
    no_steps = len(At)

    # Calculate the full correlation function
    full_corr = scp_signal.correlate(At, Bt, mode="full")
    # Normalization of the full correlation function
    norm_corr = np.array([no_steps - ii for ii in range(no_steps)])
    # Find the mid point of the array
    mid = full_corr.size // 2
    # Return only the second half of the array (positive lags)
    return full_corr[mid:] / norm_corr
