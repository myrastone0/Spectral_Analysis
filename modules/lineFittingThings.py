"""
  ~~~~~~~~~~~~~~~~~~~~
  lineFittingThings.py
  ~~~~~~~~~~~~~~~~~~~~
"""
import pyspeckit
import numpy as np
from pyspeckit.parallel_map import parallel_map


def parallelFit(contSubCube=None,pyCube=None,
                vels=None,profileMin=None,profileMax=None,
                validPixels=None,nComps=None):
  """
      Use more than one core to fit spaxels in parallel. These spaxels
      have a "high" chi^2 value. "high" is subjective here.

      Fit the individual spaxels while limiting the fitting to the
      velocity range of the line profile's location.

      Parameters
      ----------
      contSubCube: array_like
          Array containing the continuum subtracted spectra.
      pyCube: pyspeckit.Cube()
          Cube instance which contains the fitted line profile model.
      vels: array_like
          A list of the velocities along the spectral axis.
      profileMin/Max: float
          Velocities which bound the line profile range.
      validPixels: list of tuples
          Tuples are the (col,row) values of the spaxels which are to be fit.
      nComps: int
          Number of gaussian components to fit to the line profile.


      Return:
          Dictionary of the pyCube updated with the new gaussian paramters.
  """

  ## Get data shape
  nCols = contSubCube.shape[1]
  nRows = contSubCube.shape[2]

  ## Indices which bound the line profile in velocity space.
  minProfIdx = (np.abs(vels - (profileMin-75))).argmin()
  maxProfIdx = (np.abs(vels - (profileMax+75))).argmin()


  def fit_a_pixel(iicolrow):
    ii,col,row=iicolrow

    ## Make a spectral instance
    sp = pyspeckit.Spectrum(xarr=vels,
                            data=contSubCube[:,col,row],
                            unit='Jy',xarrkwargs={'unit':'km/s'})

    ## Find the peak amplitude inside the line profile range
    peakAmp = np.max(contSubCube[minProfIdx:maxProfIdx,col,row])

    ## Find the velocity of the line profile peak by computing
    ## the index of the flux closest to the peak amplitude value.
    centerIdx = (np.abs(contSubCube[:,col,row] - peakAmp)).argmin()
    gaussCenter = vels[centerIdx]

    ## Create the array for the initial guesses
    guessesArray=[peakAmp*.3,gaussCenter,50]*nComps


    ## Create some parameter limits for this spaxel
    T,F = True,False
    minMaxLimits = [((peakAmp)*.2,peakAmp),
                     (profileMin,profileMax),
                     (20,200)]*nComps

    ## Enforce the min/max parameter limits?
    minMaxLimited = [(T,T),(T,T),(T,T)]*nComps

    ## Any parameters tied to each other?
    tied = ['','','']*nComps

    try:
      ## Do the fit!
      sp.specfit(guesses=guessesArray, tied = tied,
                 show_components=True,
                 annotate= False,limits = minMaxLimits,
                 limited = minMaxLimited,quiet = True)
    except:
      pass


    return ((col,row),sp.specfit.modelpars)


  ##################################
  ## Run parallel_map for specfit ##
  ##################################
  sequence = [(ii,col,row) for ii,(col,row) in tuple(enumerate(validPixels))]
  result = parallel_map(fit_a_pixel, sequence, numcores=4)

  for i in range(len(result)):
    col,row=result[i][0][0],result[i][0][1]
    pyCube.parcube[:,col,row] = result[i][1]


  return pyCube







def parallelFitOh(contSubCube=None,pyCube=None,
                  vels=None,profileMin=None,profileMax=None,
                  validPixels=None,nComps=None):
  """
      Use more than one core to fit spaxels with line profiles in parallel.

      Fit the individual spaxels while limiting the fitting to the
      velocity range of the line profile's location.

      Parameters
      ----------
      contSubCube: array_like
          Array containing the continuum subtracted spectra.
      pyCube: pyspeckit.Cube()
          Cube instance which contains the fitted line profile model.
      vels: array_like
          A list of the velocities along the spectral axis.
      profileMin/Max: float
          Velocities which bound the line profile range.
      validPixels: list of tuples
          Tuples are the (col,row) values of the spaxels which are to be fit.
      nComps: int
          Number of gaussian components to fit to the line profile.


      Return:
          Dictionary of the pyCube updated with the new gaussian paramters.
  """

  pCygList = [1342225147,1342199415,1342237604,1342212531]

  # Get data shape
  nCols = contSubCube.shape[1]
  nRows = contSubCube.shape[2]

  # Indices which bound the line profile in velocity space.
  minProfIdx = (np.abs(vels - (profileMin-75))).argmin()
  maxProfIdx = (np.abs(vels - (profileMax+75))).argmin()


  def fit_a_pixel(iicolrow):
    ii,col,row=iicolrow

    # Make a spectral instance
    sp = pyspeckit.Spectrum(xarr=vels,
                            data=contSubCube[:,col,row],
                            unit='Jy',xarrkwargs={'unit':'km/s'})

    # Find the flux average inside of the line profile range.
    fluxAve = np.nanmean(contSubCube[minProfIdx:maxProfIdx,col,row])



    if obsId not in pCygList:
      # Absorption
      if fluxAve < 0.:
        peakAmp = np.nanmin(contSubFluxes[minProfIdx:maxProfIdx,col,row])

        # parameter limits for this spaxel
        minMaxLimits = [(peakAmp*.9,peakAmp*.1),
                         (profileMin,profileMax),
                         (50,100)]*4
        # Enforce the min/max parameter limits?
        minMaxLimited = [(T,T),(F,F),(T,T)]*4

      # Emission
      else:
        peakAmp = np.nanmax(contSubFluxes[minProfIdx:maxProfIdx,col,row])

        # parameter limits for this spaxel
        minMaxLimits = [(peakAmp*.1,peakAmp*.9),
                         (profileMin,profileMax),
                         (50,100)]*4
        # Enforce the min/max parameter limits?
        minMaxLimited = [(T,T),(F,F),(T,T)]*4

      guessesArray=[peakAmp*.5,gaussCenter+50,100,
                    peakAmp*.5,gaussCenter-50,100,
                    peakAmp*.5,gaussCenter+50+doubletSeparation,100,
                    peakAmp*.5,gaussCenter+50+doubletSeparation,100]


    # Any parameters tied to each other?
    tied = ['','','',
            '','','',
            'p[0]','p[1]+'+str(doubletSeparation),'p[2]',
            'p[3]','p[4]+'+str(doubletSeparation),'p[5]']

    try:
      ## Do the fit!
      sp.specfit(guesses=guessesArray, tied = tied,
                 show_components=True,
                 annotate= False,limits = minMaxLimits,
                 limited = minMaxLimited,quiet = True)
    except:
      pass


    return ((col,row),sp.specfit.modelpars)


  ##################################
  ## Run parallel_map for specfit ##
  ##################################
  sequence = [(ii,col,row) for ii,(col,row) in tuple(enumerate(validPixels))]
  result = parallel_map(fit_a_pixel, sequence, numcores=4)

  for i in range(len(result)):
    col,row=result[i][0][0],result[i][0][1]
    pyCube.parcube[:,col,row] = result[i][1]


  return pyCube
