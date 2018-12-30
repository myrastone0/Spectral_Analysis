"""
    ~~~~~~~~~~~~~~
    createMasks.py
    ~~~~~~~~~~~~~~
"""
import numpy as np
from numpy import ma
from scipy.signal import argrelmin



def maskTrue(data=None):
    """
        NaN values are TRUE.

        Return a dictionary of the masked
        data and the corresponding mask.

        Parameters
        ----------
        data: np.array
            Array of data.
    """
    return {'maskedData':ma.masked_invalid(data),
            'mask':ma.getmask(ma.masked_invalid(data))}


def maskFalse(data=None):
    """
        NaN values are FALSE.

        Return a dictionary of the masked
        data and the corresponding mask.

        Parameters
        ----------
        data: np.array
            Array of data.
    """

    ## Temporary mask so it can be inverted
    dataTmp = ma.masked_invalid(data)
    maskTmp = ma.getmask(dataTmp)

    maskedData = ma.masked_array(data, ~maskTmp)

    return {'maskedData':maskedData,
            'mask':ma.getmask(maskedData)}



def maskProfile(data=None,xarr=None,minProfIdx=None,maxProfIdx=None):
    """
        NaN values are TRUE.

        Mask the x-axis range where the line profile is found.
        This region needs to be ignored when fitting the continuum.

        Return a dictionary of the masked
        data and the corresponding mask.

        Parameters
        ----------
        data: np.array
            Array of data.
        vels: np.array
            Array of velocities.
        min/max: np.float
            Min and max values which bound the line profile.
    """

    ## Get the dimensions of the data cube.
    nflux, ncol, nrow = data.shape[0], data.shape[1], data.shape[2]

    ## Mask the NaN values as TRUE.
    trueMaskData=ma.masked_invalid(data)
    ## Get the mask just created.
    trueMask = ma.getmask(trueMaskData)

    ## Mask the line profile region.
    baseMaskArray = np.array(trueMask)
    for i in range(ncol):
        for j in range(nrow):
            for k in range(nflux):
                if baseMaskArray[k,i,j] != True:
                    if all([k > minProfIdx, k < maxProfIdx]):
                        baseMaskArray[k,i,j] = True
                    else:
                        baseMaskArray[k,i,j] = False

    return {'maskedData':ma.masked_array(data,baseMaskArray),
            'mask':ma.getmask(ma.masked_array(data,baseMaskArray))}


def maskEdges(data=None,xarr=None,xmin=None,xmax=None):
    """
        NaN values are TRUE.

        Mask the xarr range outside of the xarr min and max values.
        This region needs to be ignored when fitting the continuum.

        Return a dictionary of the masked
        data and the corresponding mask.

        Parameters
        ----------
        data: np.array
            Array of data.
        xarr: np.array
            Array along the spectral axis.
        min/max: np.float
            Min and max values which bound the spectrum.
    """
    ## Get the indices of the line profile edges in velocity space.

    profileMinIdx = (np.abs(xarr - (xmin))).argmin()
    profileMaxIdx = (np.abs(xarr - (xmax))).argmin()

    ## Get the dimensions of the data cube.
    nflux, ncol, nrow = data.shape[0], data.shape[1], data.shape[2]

    ## Masking the edges of the spectrum along
    ## with the line profile region.
    baseMaskArray = np.array(data)
    for i in range(ncol):
        for j in range(nrow):
            for k in range(nflux):
                if baseMaskArray[k,i,j] != True:
                    if any([k < profileMinIdx, k > profileMaxIdx]):
                        baseMaskArray[k,i,j] = True
                    else:
                        baseMaskArray[k,i,j] = False

    return {'maskedData':ma.masked_array(data,baseMaskArray),
            'mask':ma.getmask(ma.masked_array(data,baseMaskArray))}





def dataSpaxels(mask=None):
    """
        Find the indices of the spaxels which are not NaN

        Return an array of [spaxCol,spaxRow].

        Parameters
        ----------
        data: SpectralCube object
    """

    ## Get the dimensions of the data cube.
    nflux, ncol, nrow = mask.shape[0], mask.shape[1], mask.shape[2]

    goodColRow=[]
    for i in range(ncol):
        for j in range(nrow):
            if mask._mask[(nflux/2)-20:(nflux/2)+20,i,j].any() == True:
                        goodColRow.append(str([int(i),int(j)]))

    return goodColRow


def twoDSpaxMask(specCube=None,goodArray=None):
    """
        2D spaxel mask. NaN values are FALSE.

        Parameters
        ----------
        specCube: spectral cube
            Spectral cube of the data.
        goodArray: np.array
            Array of the good spaxel coordinates.
            i.e. [[col,row],[col,row]]
    """

    ## Get the dimensions of the data cube.
    ncol, nrow = specCube.shape[1], specCube.shape[2]


    twoDSpaxMask = np.zeros((ncol,nrow))
    ## Build 2D mask
    for i in range(ncol):
        for j in range(nrow):
            if str([int(i),int(j)]) in goodArray:
                twoDSpaxMask[i,j] = True
            else:
                twoDSpaxMask[i,j] = False
    return twoDSpaxMask
