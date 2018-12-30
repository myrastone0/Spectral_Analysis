"""
    Functions for velocity calculations.
"""
import numpy as np
from numpy import ma

from scipy.integrate import cumtrapz

from functionThings import *


def calcVel(fluxes=None,velMin=None,velMax=None,
            frac=None,validPixels=None,gaussParam=None,
            sigma=None,amp=None,emi=True,splineValues=None,
            velArr=None):

  """
    Compute the cumulative summation of fluxes along the velocity axis.

    (For an emission line) Find the velocity below which some
    fraction of the total integrated flux takes place.

    (E.g. v84 is the velocity below which 84% of the total integrated
    flux takes place.)

    Parameters
    ----------
    fluxes: array_like
        The 2D array of total integrated fluxes
    velMin/Max: int
        Min and max velocities
    frac: float
        Fraction of the total integrated flux.
    validPixels: array_like
        Array of (col,row) tuples of valid spaxel indices.
    gaussParam: array_like
        Parameters of the fitted gaussians of the observed line profile.
        Or, if computing the instrumental profile, leave this as None.
    emi: bool
        Is the line in emission?
    splineValues: array_like
        Continuum-subtracted fluxes that have been fitted using a spline.
    velArr: array_like
        Array of velocities that correspond to the fluxes. If None, velRange
        is computed between velMin and velMax at step=1.

    For the instrumental profile
    ----------------------------
    sigma: float
        Sigma of the instrumental profile.
    amp: float
        Amplitude of the instrumental profile.


    Return
    ------
    velocity: array_like
      Computed velocities in an array of shape fluxes.shape().
  """
  ## Get the 2D spatial dimensions
  nCols, nRows = fluxes.shape[0], fluxes.shape[1]

  ## Calculate the fraction of the total integrated flux.
  if emi==True:
    fluxFrac = abs(frac*fluxes)
  else:
    fluxFrac = frac*fluxes

  if velArr is None:
      ## Create a velocity range which increases by 1.
      velRange = np.arange(velMin,velMax,1)
      nFluxes = len(velRange)
  else:
      velRange = velArr
      nFluxes = len(velArr)


  ## Array to hold the computed velocities
  velArray=np.zeros((nCols,nRows))

  ## Calculate the fluxes of the gaussian
  ## (or, if multiple gaussians, the sum of the gaussians).
  lineFunc = np.zeros((nFluxes,nCols,nRows))
  for coord in validPixels:
    col, row = coord[0], coord[1]

    if gaussParam is not None:
        ## Two gaussian components
        if len(gaussParam[:,0,0])==6:
          lineFunc[:,col,row] = gaussFunc(velRange,gaussParam[0:3,col,row])\
                              + gaussFunc(velRange,gaussParam[3:6,col,row])
        ## Three gaussian components
        if len(gaussParam[:,0,0]) == 9:
          lineFunc[:,col,row] = gaussFunc(velRange,gaussParam[0:3,col,row])\
                              + gaussFunc(velRange,gaussParam[3:6,col,row])\
                              + gaussFunc(velRange,gaussParam[6:9,col,row])

        ## Four gaussian components
        if len(gaussParam[:,0,0]) == 12:
          lineFunc[:,col,row] = gaussFunc(velRange,gaussParam[0:3,col,row])\
                              + gaussFunc(velRange,gaussParam[3:6,col,row])\
                              + gaussFunc(velRange,gaussParam[6:9,col,row])\
                              + gaussFunc(velRange,gaussParam[9:12,col,row])

    if splineValues is not None:
        lineFunc[:,col,row] = splineValues[:,col,row]

    ## Calculate the cumulatively integrated flux at each velocity value.
    trapz=cumtrapz(lineFunc[:,col,row],velRange,initial=0)

    ## Get the index of the velocity which is closest to
    ## the desired fraction of the total integrated velocity.
    velIdx = (np.abs(trapz - (fluxFrac[col,row]))).argmin()

    ## Populate the velocity array with the computed velocities.
    velArray[col,row] = velRange[velIdx]

  return velArray







def calcVelWidth(velDict):
  """
    Calculate the 1sigma and 2sigma widths of a line profile.

    Also calculates a measure of the line's asymmetry via:
        abs(v50-v84) + (v16-v50),
    where v50 is the velocity below which 50% of the emission takes place.

    A negative value of asymmetry indicates that the line profile is more
    skewed to the blue and a positive value indicates that the line profile
    is more skewed to the red end of the spectrum.


    Parameters
    ----------
    velDict: dictionary
        Dictionary of the computed velocites. The format is:
            {'v02':{'frac':0.023,'image':[]}}
        where 'frac' indicates the fraction of the total integrated flux
        to compute and 'image' is a 2D array of the computed velocities.

    Return
    ------
    List containing the images of w1, w2, and wAsym.

  """

  w1 = velDict['v84']['image'] - velDict['v16']['image']
  w2 = velDict['v97']['image'] - velDict['v02']['image']
  wAsym = np.abs(velDict['v50']['image']
                -velDict['v84']['image'])\
          + (velDict['v16']['image']\
             -velDict['v50']['image'])

  return [w1,w2,wAsym]




def calcVelOh(gaussParams=None,velMin=None,velMax=None,mask=None,
              fluxDict=None,validPixels=None):

  nCols,nRows = gaussParams.shape[1],gaussParams.shape[2]
  totalIntFluxAbs = fluxDict['totalAbs']
  totalIntFluxEmi = fluxDict['totalEmi']
  totalIntFlux = fluxDict['totalFlux']

  ## For absorption
  velDictAbs = {'v01':{'frac':0.998,'image':[]},
                'v02':{'frac':0.977,'image':[]},
                'v16':{'frac':0.841,'image':[]},
                'v50':{'frac':0.500,'image':[]},
                'v84':{'frac':0.156,'image':[]},
                'v98':{'frac':0.023,'image':[]},
                'v99':{'frac':2e-3,'image':[]}
               }
  for key in velDictAbs:
    absVels = np.arange(velMin,velMax,1)
    numAbsFluxes = len(absVels)

    absVelArray = np.zeros((nCols,nRows))
    lineFunc = np.zeros((numAbsFluxes,nCols,nRows))

    for coord in validPixels:
      col,row=coord[0],coord[1]

      if all([gaussParams[0,col,row] < 0,gaussParams[3,col,row] > 0]):
        fluxVel = totalIntFluxAbs[col,row]*velDictAbs[key]['frac']
        lineFunc[:,col,row] = gaussFunc(absVels,gaussParams[0:3,col,row])\
                            + gaussFunc(absVels,gaussParams[6:9,col,row])
      elif all([gaussParams[0,col,row] > 0,gaussParams[3,col,row] < 0]):
        fluxVel = totalIntFluxAbs[col,row]*velDictAbs[key]['frac']
        lineFunc[:,col,row] = gaussFunc(absVels,gaussParams[3:6,col,row])\
                            + gaussFunc(absVels,gaussParams[9:12,col,row])
      else:
        fluxVel = totalIntFlux[col,row]*velDictAbs[key]['frac']
        lineFunc[:,col,row] = gaussFunc(absVels,gaussParams[0:3,col,row])\
                            + gaussFunc(absVels,gaussParams[3:6,col,row])\
                            + gaussFunc(absVels,gaussParams[6:9,col,row])\
                            + gaussFunc(absVels,gaussParams[9:12,col,row])

      trapz=cumtrapz(lineFunc[:,col,row],absVels,initial=0)
      velAbsIdx = (np.abs(trapz - (fluxVel))).argmin()
      absVelArray[col,row] = absVels[velAbsIdx]

    ## Do some masking.
    velDictAbs[key]['image'] = ma.masked_array(absVelArray,mask)

    # velDictAbs[key]['image'] = ma.masked_array(velDictAbs[key]['image'],
    #                                            ma.getmask(totalIntFluxAbs))


  ## For emission
  velDictEmi = {'v002':{'frac':2e-3,'image':[]},
                'v02':{'frac':0.023,'image':[]},
                'v16':{'frac':0.156,'image':[]},
                'v50':{'frac':0.500,'image':[]},
                'v84':{'frac':0.841,'image':[]},
                'v97':{'frac':0.977,'image':[]},
                'v99':{'frac':0.998,'image':[]}
               }
  for key in velDictEmi:
    emiVels = np.arange(velMin,velMax,1)
    numEmiFluxes = len(emiVels)

    emiVelArray = np.zeros((nCols,nRows))
    lineFunc = np.zeros((numEmiFluxes,nCols,nRows))

    for coord in validPixels:
      col,row=coord[0],coord[1]

      if all([gaussParams[0,col,row] < 0,gaussParams[3,col,row] > 0]):
        fluxVel = abs(totalIntFluxEmi[col,row]*velDictEmi[key]['frac'])
        lineFunc[:,col,row] = gaussFunc(emiVels,gaussParams[3:6,col,row])\
                            + gaussFunc(emiVels,gaussParams[9:12,col,row])
      elif all([gaussParams[0,col,row] > 0,gaussParams[3,col,row] < 0]):
        fluxVel = abs(totalIntFluxEmi[col,row]*velDictEmi[key]['frac'])
        lineFunc[:,col,row] = gaussFunc(emiVels,gaussParams[0:3,col,row])\
                            + gaussFunc(emiVels,gaussParams[6:9,col,row])
      else:
        fluxVel = abs(totalIntFlux[col,row]*velDictEmi[key]['frac'])
        lineFunc[:,col,row] = gaussFunc(emiVels,gaussParams[0:3,col,row])\
                            + gaussFunc(emiVels,gaussParams[3:6,col,row])\
                            + gaussFunc(emiVels,gaussParams[6:9,col,row])\
                            + gaussFunc(emiVels,gaussParams[9:12,col,row])

      trapz=cumtrapz(lineFunc[:,col,row],emiVels,initial=0)
      velEmiIdx = (np.abs(trapz - (fluxVel))).argmin()
      emiVelArray[col,row] = emiVels[velEmiIdx]

    ## Do some masking.
    velDictEmi[key]['image'] = ma.masked_array(emiVelArray,
                                               mask)

    # velDictEmi[key]['image'] = ma.masked_where(velDictEmi[key]['image']==0,
    #                                            velDictEmi[key]['image'])

    # velDictEmi[key]['image'] = ma.masked_array(velDictEmi[key]['image'],
    #                                            ma.getmask(totalIntFluxEmi))


  return {'velAbs' : velDictAbs,
          'velEmi' : velDictEmi}


