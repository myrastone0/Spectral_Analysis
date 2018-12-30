"""
  ~~~~~~~~~~~~~
  fluxThings.py
  ~~~~~~~~~~~~~
"""
from scipy.integrate import quad
import numpy as np
from numpy import ma

from functionThings import *




def calcGaussFluxes(gaussParams=None,validPixels=None,velMin=None,velMax=None,
                    interval=1):
  """
    Calculate the fluxes of a gaussian component at each wavelength.

    Parameters
    ----------
    gaussParam: array_like
      List of the gaussian parameters (amplitude,peak velocity,sigma).
      This is for a SINGLE gaussian and is a 3D array where
      the parameters are in the 0-axis.
    vels: array_like
      List of velocities for which the fluxes are to be computed.
    validPixels: array_like
      List of (col,row) tuples of the spaxels which are valid.
    interval: int
      Interval between velocities in the velocities array.

    Return
    ------
    gaussFluxes: array_like
      Array of fluxes computed at each velocity in vels.
  """
  vels = np.arange(velMin,velMax,interval)

  nCols, nRows = gaussParams.shape[1], gaussParams.shape[2]
  nFluxes = len(vels)

  ## Empty array which will hold the fluxes.
  gaussFluxes = np.zeros((nFluxes,nCols,nRows))

  ## Compute the fluxes only for pixels which are valid.
  for coord in validPixels:
    col,row = coord[0], coord[1]
    for ii in range(len(vels)):
      gaussFluxes[ii,col,row] = gaussFunc(vels[ii],gaussParams[:,col,row])

  return gaussFluxes




def calcTotIntFluxes(gaussParams=None,validPixels=None,
                     velMin=None,velMax=None,xValues=None):
  """
    Compute the total integrated flux for each valid pixel in an image.

    Parameters
    ----------
    gaussParams: array_like
      Array of gaussian parameters. This can handle multiple gaussian components.
    validPixels: array_like
      List of (col,row) tuples of the spaxels which are valid.
    velMin/Max: int
      Minimum and maximum velocities of the integral.

    Return
    ------
    integralMap: array_like
      Array of the integrated fluxes.

  """
  nCols,nRows = gaussParams.shape[1],gaussParams.shape[2]
  integralMap = np.zeros((nCols,nRows))

  if velMin != None:
    xArr = np.arange(velMin,velMax,1)
  if velMin == None:
    xArr = xValues



  for coord in validPixels:
    col,row = coord[0],coord[1]

    ## 4 guassian components
    if len(gaussParams) == 12:
      totalIntFlux = quad(lambda x:
                          gaussFunc(x,gaussParams[0:3,col,row]) +\
                          gaussFunc(x,gaussParams[3:6,col,row]) +\
                          gaussFunc(x,gaussParams[6:9,col,row]) +\
                          gaussFunc(x,gaussParams[9:12,col,row]),
                          xArr[0], xArr[-1]+1)
    ## 3 guassian components
    if len(gaussParams) == 9:
      totalIntFlux = quad(lambda x:
                          gaussFunc(x,gaussParams[0:3,col,row]) +\
                          gaussFunc(x,gaussParams[3:6,col,row]) +\
                          gaussFunc(x,gaussParams[6:9,col,row]),
                          xArr[0], xArr[-1]+1)
    ## 2 guassian components
    if len(gaussParams) == 6:
      totalIntFlux = quad(lambda x:
                          gaussFunc(x,gaussParams[0:3,col,row]) +\
                          gaussFunc(x,gaussParams[3:6,col,row]),
                          xArr[0], xArr[-1]+1)

    integralMap[col,row] = totalIntFlux[0]

  return integralMap





def makeFluxSigMask(flux=None,minThresh=2,maxThresh=5):
    """
      Compute the mean total integrated flux value
      and its standard deviation.

      Find all pixels with a flux in between min/max thresholds and mask them.

      Parameters
      ----------
      fluxImage: array_like
        The 2D array of the total integrated fluxes.
      min/maxThresh: int
        Sigma limit thresholds for the min/max.

      Return
      ------
      Boolean mask.
    """

    sigma = ma.std(flux)
    ave = ma.mean(flux)

    if sigma > ave:
      intervalMin = ave
    else:
      intervalMin = ave-(minThresh*sigma)

    intervalMax = ave+(maxThresh*sigma)

    maskedOutside = ma.masked_outside(flux,intervalMin,intervalMax)
    maskedZeros = ma.masked_where(maskedOutside==0,maskedOutside,copy=False)

    return ma.getmask(maskedZeros)




def calcTotIntFluxesOh(gaussParams=None,validPixels=None,
                       velMin=None,velMax=None,mask=None):
  """
    Compute the total integrated flux for each valid pixel
    in an image for the OH observations.

    Parameters
    ----------
    gaussParams: array_like
      Array of gaussian parameters. This can handle multiple gaussian components.
    validPixels: array_like
      List of (col,row) tuples of the spaxels which are valid.
    velMin/Max: int
      Minimum and maximum velocities of the integral.
    mask: array_like
      Boolean array where good data is FALSE.

    Return
    ------
    integralMap: dict
      Dictionary the pure absorption, pure emission,
      and/or pCyg integrated fluxes.

  """
  nCols,nRows = gaussParams.shape[1],gaussParams.shape[2]
  totalIntFluxAbs = np.zeros((nCols,nRows))
  totalIntFluxEmi = np.zeros((nCols,nRows))
  totalIntFlux = np.zeros((nCols,nRows))

  # Create a list of velocities for integration
  xArr = np.arange(velMin,velMax,1)


  for coord in validPixels:
    col,row = coord[0], coord[1]

    # Pure absorption
    if all([gaussParams[0,col,row]<0, gaussParams[3,col,row]<0]):
      totalIntFluxAbs[col,row] = quad(lambda x:
                                 gaussFunc(x,gaussParams[0:3,col,row]) +\
                                 gaussFunc(x,gaussParams[3:6,col,row]) +\
                                 gaussFunc(x,gaussParams[6:9,col,row]) +\
                                 gaussFunc(x,gaussParams[9:12,col,row]),
                                 xArr[0], xArr[-1]+1)[0]
    # Pure emission
    if all([gaussParams[0,col,row]>0, gaussParams[3,col,row]>0]):
      totalIntFluxEmi[col,row] = quad(lambda x:
                                      gaussFunc(x,gaussParams[0:3,col,row]) +\
                                      gaussFunc(x,gaussParams[3:6,col,row]) +\
                                      gaussFunc(x,gaussParams[6:9,col,row]) +\
                                      gaussFunc(x,gaussParams[9:12,col,row]),
                                      xArr[0], xArr[-1]+1)[0]

    # Normal pCyg
    if all([gaussParams[0,col,row] < 0,gaussParams[3,col,row] > 0]):
      pCyg=True
      totalIntFluxAbs[col,row] = quad(lambda x:
                                  gaussFunc(x,gaussParams[0:3,col,row])+\
                                  gaussFunc(x,gaussParams[6:9,col,row]),
                                  xArr[0], xArr[-1]+1)[0]
      totalIntFluxEmi[col,row] = quad(lambda x:
                                  gaussFunc(x,gaussParams[3:6,col,row])+\
                                  gaussFunc(x,gaussParams[9:12,col,row]),
                                  xArr[0], xArr[-1]+1)[0]

    # Inverse pCyg
    if all([gaussParams[0,col,row] > 0,gaussParams[3,col,row] < 0]):
      pCyg=True
      totalIntFluxAbs[col,row] = quad(lambda x:
                                  gaussFunc(x,gaussParams[3:6,col,row])+\
                                  gaussFunc(x,gaussParams[9:12,col,row]),
                                  xArr[0], xArr[-1]+1)[0]
      totalIntFluxEmi[col,row] = quad(lambda x:
                                  gaussFunc(x,gaussParams[0:3,col,row])+\
                                  gaussFunc(x,gaussParams[6:9,col,row]),
                                  xArr[0], xArr[-1]+1)[0]


    # Get the sum of both the abs and emi fluxes.
    # This will only be different if the spaxel contains a pCyg profile.
    totalIntFlux[col,row] = quad(lambda x:
                             gaussFunc(x,gaussParams[0:3,col,row])+\
                             gaussFunc(x,gaussParams[3:6,col,row])+\
                             gaussFunc(x,gaussParams[6:9,col,row])+\
                             gaussFunc(x,gaussParams[9:12,col,row]),
                             xArr[0], xArr[-1]+1)[0]


    # Mask the total integrated fluxes
    totalIntFlux = ma.masked_array(totalIntFlux,mask)
    totalIntFluxAbs = ma.masked_array(totalIntFluxAbs,mask)
    totalIntFluxEmi = ma.masked_array(totalIntFluxEmi,mask)

    totalIntFlux = ma.masked_where(totalIntFlux==0,totalIntFlux)
    totalIntFluxAbs = ma.masked_where(totalIntFluxAbs==0,totalIntFluxAbs)
    totalIntFluxEmi = ma.masked_where(totalIntFluxEmi==0,totalIntFluxEmi)

  return {'totalAbs'  : totalIntFluxAbs,
          'totalEmi'  : totalIntFluxEmi,
          'totalFlux' : totalIntFlux}
