import numpy as np
from scipy import optimize
from typing import List, Tuple, Optional, Union


class fit:

    def doubleexp(self, x: np.ndarray, A: float, alpha: float, tau1: float, tau2: float) -> np.ndarray:
        return A * alpha * tau1 * (1 - np.exp(-x / tau1)) + A * (1 - alpha) * tau2 * (1 - np.exp(-x / tau2))

    def singleexp(self, x: np.ndarray, A: float, tau: float) -> np.ndarray:
        return A * (1 - np.exp(-x / tau))

    def powerlaw_std(self, t: np.ndarray, A: float, B: float) -> np.ndarray:
        return A * t**B

    def fit(
        self,
        time: np.ndarray,
        ave: np.ndarray,
        stddev: np.ndarray,
        use_double_exp: bool,
        popt2: List[float],
        std_perc: Optional[float] = None,
        endt: Optional[float] = None,
    ) -> Tuple[float, np.ndarray, int]:
        """
        Fit data with single or double exponential function.

        Parameters:
            time: Time array
            ave: Average values to fit
            stddev: Standard deviation of values
            use_double_exp: Whether to use double exponential fit
            popt2: Initial guess parameters for fitting
            std_perc: Standard deviation percentage for cutoff
            endt: End time for fitting

        Returns:
            Tuple containing:
                - Fitted value
                - Fitted curve
                - Cut index
        """
        foundcutoff = False
        start = 1
        cut = start + 1

        popt_power, pcov_power = optimize.curve_fit(self.powerlaw_std, time, stddev, maxfev=100000)
        stddev_fit = self.powerlaw_std(time, *popt_power)

        if endt is None:
            if std_perc is None:
                std_perc = 0.4
            while not foundcutoff and cut < len(ave):
                if stddev_fit[cut] > std_perc * ave[cut]:
                    foundcutoff = True
                else:
                    cut += 1
        else:
            diff = np.abs(time - endt)
            cut = np.argmin(diff)

        if use_double_exp:
            # Use double exponential fit
            popt2, pcov2 = optimize.curve_fit(
                self.doubleexp,
                time[start:cut],
                ave[start:cut],
                maxfev=1000000,
                p0=popt2,
                sigma=stddev[start:cut],
                bounds=(0, [np.inf, 1, np.inf, np.inf]),
            )
            fit = self.doubleexp(time, *popt2)
            Value = popt2[0] * popt2[1] * popt2[2] + popt2[0] * (1 - popt2[1]) * popt2[3]
        else:
            # Use single exponential fit
            popt2, pcov2 = optimize.curve_fit(
                self.singleexp,
                time[start:cut],
                ave[start:cut],
                maxfev=1000000,
                p0=popt2,
                sigma=stddev[start:cut],
                bounds=(0, [np.inf, np.inf]),
            )
            fit = self.singleexp(time, *popt2)
            Value = popt2[0]

        return Value, fit, cut
