import scipy.signal as scp_signal
import numpy as np


def correlationfunction(At, Bt):
    """
    Calculate the correlation function between :math:`\\mathbf{A}(t)` and :math:`\\mathbf{B}(t)` using
    :func:`scipy.signal.correlate`

    .. math::
        C_{AB}(\\tau) =  \\sum_j^D \\sum_i^T A_j(t_i)B_j(t_i + \\tau)

    where :math:`D` is the number of dimensions and :math:`T` is the total length
    of the simulation.

    Parameters
    ----------
    At : numpy.ndarray
        Observable to correlate.

    Bt : numpy.ndarray
        Observable to correlate.

    Returns
    -------
    full_corr : numpy.ndarray
        Correlation function :math:`C_{AB}(\\tau)`

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace( 0.0, 6.0 * np.pi, 3000)
    >>> w0 = 0.5
    >>> At = np.cos(w0 * t)
    >>> Bt = np.sin(w0 * t)
    >>> corr_t = correlationfunction(At, Bt)

    """
    no_steps = At.size

    # Calculate the full correlation function.
    full_corr = scp_signal.correlate(At, Bt, mode="full")
    # Normalization of the full correlation function, Similar to norm_counter
    norm_corr = np.array([no_steps - ii for ii in range(no_steps)])
    # Find the mid point of the array
    mid = full_corr.size // 2
    # I want only the second half of the array, i.e. the positive lags only
    return full_corr[mid:] / norm_corr
