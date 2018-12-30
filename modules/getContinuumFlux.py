"""
    ~~~~~~~~~~~~~~~~~~~
    getContinuumFlux.py
    ~~~~~~~~~~~~~~~~~~~
From `agpy <http://code.google.com/p/agpy/source/browse/trunk/agpy/cubes.py>`_,
contains functions to perform various transformations on data cubes and their
headers.
"""

from astropy.extern.six.moves import xrange
from numpy import sqrt,repeat,indices,newaxis,pi,cos,sin,array,mean,nansum
from math import acos,atan2,tan
import numpy
import numpy as np
import copy
import os
import astropy.io.fits as fits
import astropy.wcs as pywcs
import tempfile
import warnings
import astropy
from astropy import coordinates
from astropy import log
try:
    from AG_fft_tools import smooth
    smoothOK = True
except ImportError:
    smoothOK = False
try:
    from scipy.interpolate import UnivariateSpline
    scipyOK = True
except ImportError:
    scipyOK = False


from pyspeckit.parallel_map import parallel_map

dtor = pi/180.0


def blfunc_generator(x=None, polyorder=None, splineorder=None,
                     sampling=1):
    """
    Generate a function that will fit a baseline (polynomial or spline) to a
    data set.  Either ``splineorder`` or ``polyorder`` must be set
    Parameters
    ----------
    x : np.ndarray or None
        The X-axis of the fitted array.  Will be set to
        ``np.arange(len(data))`` if not specified
    polyorder : None or int
        The polynomial order.
    splineorder : None or int
    sampling : int
        The sampling rate to use for the data.  Can set to higher numbers to
        effectively downsample the data before fitting
    """
    def blfunc(args, x=x):
        yfit,yreal = args
        if hasattr(yfit,'mask'):
            mask = ~yfit.mask
        else:
            mask = np.isfinite(yfit)

        if x is None:
            x = np.arange(yfit.size, dtype=yfit.dtype)

        ngood = np.count_nonzero(mask)
        if polyorder is not None:
            if ngood < polyorder:
                return yreal
            else:
                endpoint = ngood - (ngood % sampling)
                y = np.mean([yfit[mask][ii:endpoint:sampling]
                             for ii in range(sampling)], axis=0)
                polypars = np.polyfit(x[mask][sampling/2:endpoint:sampling],
                                      y, polyorder)
                return np.polyval(polypars, x).astype(yreal.dtype)

        elif splineorder is not None and scipyOK:
            if splineorder < 1 or splineorder > 4:
                raise ValueError("Spline order must be in {1,2,3,4}")
            elif ngood <= splineorder:
                return yreal
            else:
                log.debug("splinesampling: {0}  "
                          "splineorder: {1}".format(sampling, splineorder))
                endpoint = ngood - (ngood % sampling)
                y = np.mean([yfit[mask][ii:endpoint:sampling]
                             for ii in range(sampling)], axis=0)
                if len(y) <= splineorder:
                    raise ValueError("Sampling is too sparse.  Use finer sampling or "
                                     "decrease the spline order.")
                spl = UnivariateSpline(x[mask][sampling/2:endpoint:sampling],
                                       y,
                                       k=splineorder,
                                       s=0)
                return yreal-spl(x)
        else:
            raise ValueError("Must provide polyorder or splineorder")

    return blfunc


def getContinuumFlux(cube, polyorder=None, cubemask=None, splineorder=None,
                  numcores=None, sampling=1,xarr=None):
    """
    Given a cube, fit a polynomial to each spectrum
    Parameters
    ----------
    cube: np.ndarray
        An ndarray with ndim = 3, and the first dimension is the spectral axis
    polyorder: int
        Order of the polynomial to fit and subtract
    cubemask: boolean ndarray
        Mask to apply to cube.  Values that are True will be ignored when
        fitting.
    numcores : None or int
        Number of cores to use for parallelization.  If None, will be set to
        the number of available cores.
    """
    x = xarr
    #polyfitfunc = lambda y: np.polyfit(x, y, polyorder)
    blfunc = blfunc_generator(x=x,
                              splineorder=splineorder,
                              polyorder=polyorder,
                              sampling=sampling)

    reshaped_cube = cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2]).T

    if cubemask is None:
        log.debug("No mask defined.")
        fit_cube = reshaped_cube
    else:
        if cubemask.dtype != 'bool':
            raise TypeError("Cube mask *must* be a boolean array.")
        if cubemask.shape != cube.shape:
            raise ValueError("Mask shape does not match cube shape")
        log.debug("Masking cube with shape {0} "
                  "with mask of shape {1}".format(cube.shape, cubemask.shape))
        masked_cube = cube.copy()
        masked_cube[cubemask] = np.nan
        fit_cube = masked_cube.reshape(cube.shape[0], cube.shape[1]*cube.shape[2]).T


    baselined = np.array(parallel_map(blfunc, zip(fit_cube,reshaped_cube), numcores=numcores))
    blcube = baselined.T.reshape(cube.shape)
    return blcube
