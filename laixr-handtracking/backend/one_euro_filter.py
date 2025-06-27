import math

class LowPassFilter:
    def __init__(self, alpha):
        self.__set_alpha(alpha)
        self.__y = None
        self.__s = None

    def __set_alpha(self, alpha):
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):
        if alpha is not None:
            self.__set_alpha(alpha)
        if self.__y is None:
            self.__s = value
        else:
            self.__s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        return self.__s

    def last(self):
        return self.__s

class OneEuroFilter:
    """
    A one-euro filter for smoothing signals.
    This implementation is based on the C++ version by aex_ and the Python version by G-T-R.
    https://gist.github.com/G-T-R/1816d0c75c5e3176b6a0
    """
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq is None:
            raise ValueError("freq is None")
        if mincutoff is None:
            raise ValueError("mincutoff is None")
        if beta is None:
            raise ValueError("beta is None")
        if dcutoff is None:
            raise ValueError("dcutoff is None")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None

    def __alpha(self, cutoff):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, timestamp=None):
        if self.__lasttime is not None and timestamp is not None:
            dt = timestamp - self.__lasttime
            if dt > 0:
                self.__freq = 1.0 / dt
        self.__lasttime = timestamp
        
        prev_x = self.__x.last()
        if prev_x is None:
            dx = 0.0
        else:
            dx = (x - prev_x) * self.__freq
            
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        cutoff = self.__mincutoff + self.__beta * abs(edx)
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff)) 