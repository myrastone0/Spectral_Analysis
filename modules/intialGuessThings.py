"""

    intialGuessThings.py

"""
import numpy as np


def buildGuessCube((nCols,nRows),contSubCube=None,vels=None,
                   validPixels=None,objectName=None,minIdx=None,maxIdx=None):
    """
      Build a cube of initial guesses for the line profile fit.

      Parameters
      ----------
      (nCols,nRows): array_like
        2D shape of the spatial pixels.
      contSubCube: array_like
        3D array of the continuum subtracted spectra.
      vels: array_like
        List of velocities along the spectral axis.
      validPixels: array_like
        List of (col,row) values of valid pixels.
      objectName: str
        The object's name.
      min/maxIdx: int
        Minimum and maximum index values which bound the range of the line profile.

      Return
      ------
      [3D array of initial guesses,2D array of the peak fluxes]

    """

    ## Arrays to hold the gaussian center and gaussian amplitude guesses.
    gaussCenter = np.zeros((nCols,nRows))
    gaussAmp = np.zeros((nCols,nRows))

    for coord in validPixels:
        col,row = coord[0],coord[1]

        ## Find the peak amplitude inside the line profile range
        peakAmp = np.nanmax(contSubCube[minIdx:maxIdx,col,row])
        gaussAmp[col,row] = peakAmp

        ## Find the velocity of the line profile peak by computing
        ## the index of the flux closest to the peak amplitude value.
        centerIdx = (np.abs(contSubCube[:,col,row] - peakAmp)).argmin()
        gaussCenter[col,row] = vels[centerIdx]

    ## Create an image of a sigma guess. 50 km/s is a good guess.
    gaussSigma = np.full((nCols, nRows), 50)

    ## Build the guesses cube for fitting the line profile.
    if objectName == 'ngc1068':
      guessCube=np.array([gaussAmp*.3,gaussCenter-(gaussSigma*.2),gaussSigma,
                          gaussAmp*.3,gaussCenter,gaussSigma,
                          gaussAmp*.3,gaussCenter+(gaussSigma*.2),gaussSigma])
    else:
      guessCube=np.array([gaussAmp*.4,gaussCenter-(gaussSigma*.2),gaussSigma,
                         gaussAmp*.4,gaussCenter+(gaussSigma*.2),gaussSigma])

    return [guessCube,gaussAmp]
