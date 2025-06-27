import math

class OneEuroFilter:
    """
    A simple implementation of the 1€ Filter.
    This filter is robust to jitter and lag, using a first-order low-pass
    filter with an adaptive cutoff frequency.
    """
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        #--- The filtered derivative of the signal
        # Update the sampling rate
        if t_e > 1e-6:
            freq = 1.0 / t_e
        else:
            freq = 1e6 # A large number to avoid division by zero
        
        # Estimate the derivative
        dx = (x - self.x_prev) / t_e if t_e > 1e-6 else 0.0
        
        # Filter the derivative
        a_d = self._smoothing_factor(freq, self.d_cutoff)
        self.dx_prev = self._exponential_smoothing(a_d, dx, self.dx_prev)

        #--- The filtered signal
        # Compute the adaptive cutoff frequency
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        
        # Filter the signal
        a = self._smoothing_factor(freq, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values
        self.x_prev = x_hat
        self.t_prev = t

        return x_hat

    def _smoothing_factor(self, freq, cutoff):
        r = 2 * math.pi * cutoff / freq
        return r / (r + 1)

    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev 