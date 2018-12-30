"""
    This is a script which utilizes the ObsInfo class.

    Create an instance of ObsInfo using data and parameters from a single Herschel
    observation.

    Fit a polynomial to a spectral continuum.

    Fit a line profile with 2 (or 3, if NGC 1068) Gaussian components.

    Measure line profile properties from the fits.

    Correct widths at 1 and 2 sigma for instrumental resolution.

    Plot the continuum fits.

    Plot the line profile properties as colormaps.
"""

# ------- #
# Imports #
# ------- #
# Standard
import os
import numpy as np
import time
# Related
import pyspeckit
from pyspeckit import cubes
from pyspeckit.spectrum.parinfo import *
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from spectral_cube import SpectralCube, BooleanArrayMask
# Local
from plotThings import plotSpectra, plotMultiMap
from fitsThings import buildWcs, createHdu
from maskThings import maskTrue, maskFalse, maskProfile
from getContinuumFlux import getContinuumFlux
from pixelThings import findValidPixels, pixMinMax
from intialGuessThings import buildGuessCube
from fluxThings import calcGaussFluxes, calcTotIntFluxes
from velocityThings import calcVel, calcVelWidth
from lineFittingThings import parallelFit
from dictionaryThings import loadDict
from ObsInfo import ObsInfo



# Size of the interpolated spaxels
arcsec = '6arc'
# Number of Gaussian components to fit to the line profile.
nComps = 2

# ----------------------------------------- #
# Necessary parameter file and dictionaries #
# ----------------------------------------- #
# Dictionary of galaxy properties
objDictName = 'objectInfoDict.pkl'
# Dictionary of emission line properties.
lineDict = loadDict('emiLineDict.pkl')
# Text file of line fitting parameters.
paramFileName = 'fittingParameters.txt'


# -------------------------------------------------- #
# Read in the parameter file containing line profile #
# velocity limits and continuum fitting information. #
# -------------------------------------------------- #
paramFileData = np.genfromtxt(paramFileName, dtype = None,
                              autostrip = True, names = True, encoding=None)


for x in range(len(paramFileData)):
    # ------------------------------------ #
    # Get the galaxy and line information. #
    # ------------------------------------ #
    obsInfo = ObsInfo(x, paramFileName, objDictName)

    # Name of the emission line.
    lineName = paramFileData['lineNameShort'][x]
    # Rest wavelength of the emission line.
    restWave = lineDict[lineName]['restWave']

    # Order of the polynomial to be fit to the continuum.
    polyorder = obsInfo.polyorder


    # ---------------------------------- #
    # Directory and Labeling Name Bases  #
    # ---------------------------------- #
    # Base for labeling inside plots.
    objectLabel = (obsInfo.objectName.upper()+'  '+str(obsInfo.obsId)+'  '
                   +lineDict[lineName]['texLabel'])
    # Base for the object's file names.
    objectNameBase = (str(obsInfo.obsId)+'_'+obsInfo.objectName+'_'+lineName)
    # Base path to the object's directories.
    objectPathBase = (obsInfo.objectName+'/')
    if (not os.path.exists(objectPathBase)):os.makedirs(objectPathBase)


    # --------------------------------- #
    # Paths to save PDF and FITS files. #
    # --------------------------------- #
    # Continuum plots
    if arcsec != '1arc':
        contSavePath = objectPathBase + 'contPlots/'
        if (not os.path.exists(contSavePath)):os.makedirs(contSavePath)
    # Line Property Maps
    mapSavePath = objectPathBase + 'propertyMaps/'
    if (not os.path.exists(mapSavePath)):os.makedirs(mapSavePath)
    # Output FITS files
    fitsSavePath = objectPathBase + 'outFitsFiles/'
    if (not os.path.exists(fitsSavePath)):os.makedirs(fitsSavePath)


    # ---------------------------------------------- #
    # Open and read in FITS file of the observation. #
    # ---------------------------------------------- #
    fitsFile = (objectNameBase+'_Hipe_15_0_3244_eqInter'+arcsec+'.fits')
    fitsHdu = fits.open(fitsFile)

    # Convert spectral axis wavelengths to velocities and correct for redshift.
    obsInfo.computeVels(fitsFile, restWave, velCorr=True)

    # Crop the fluxes and velocities using velMin and velMax from parameter file.
    vels = obsInfo.velsCorr[obsInfo.minIdx:obsInfo.maxIdx]
    data = obsInfo.originalHdu['image'].data[obsInfo.minIdx:obsInfo.maxIdx,:,:]

    # Number of rows, columns, and fluxes in the cropped 3D data array.
    nRows,nCols,nFluxes = data.shape[2], data.shape[1], data.shape[0]


    # ------------- #
    # Build the WCS #
    # ------------- #
    # The spectral axis is corrected for redshift in buildWcs function.
    pacsWcs=buildWcs(file=fitsFile, restWave=restWave, z=obsInfo.z)


    # ---------------- #
    # Build the masks. #
    # ---------------- #
    # The masked data and their corresponding masks are returned as dictionaries.
    # Mask where NaNs are TRUE
    trueMask = maskTrue(data=data)
    # Mask where NaNs are FALSE
    falseMask = maskFalse(data=data)
    # Mask the profile range. NaNs are TRUE
    profileMask = maskProfile(data=trueMask['maskedData'],xarr=vels,
                              minProfIdx=obsInfo.minProfIdx,maxProfIdx=obsInfo.maxProfIdx)
    # Build mask with a WCS that is compatible with SpectralCube.
    # NaNs are FALSE.
    cubeMask = BooleanArrayMask(falseMask['mask'],pacsWcs)

    # Make an array of coordinates for the valid spaxels
    validPixels = findValidPixels((nCols,nRows),trueMask['mask'][0,:,:])


    # --------------------------------------------------- #
    # Find the (col,row) min/max limits of valid spaxels. #
    # Min/max will be used to crop the data and the WCS.  #
    # --------------------------------------------------- #
    wcsMinMax = pixMinMax(validPixels)


    # ------------------------------------------------------------- #
    # Fit the continuum with a polynomial and subtract from fluxes. #
    # ------------------------------------------------------------- #
    # The edges of the spectrum and the velocity region
    # of the line profile is masked as TRUE. Data which
    # is to be fit as the continuum is masked as FALSE.
    # Flux of the fitted continuum
    contFluxCube=getContinuumFlux(profileMask['maskedData'],
                                  polyorder = polyorder,
                                  cubemask = profileMask['mask'],
                                  numcores = 4,
                                  sampling = 1,
                                  xarr = vels)
    # Mask NaNs
    contFluxesMasked = np.ma.masked_array(contFluxCube,mask=trueMask['mask'])

    # Subtract the continuum from the spectra.
    # Line profile range is not masked. NaNs are masked as TRUE.
    contSubCube = trueMask['maskedData'] - contFluxCube


    # --------------------------------------------- #
    # Build the spectral cube using the contSubCube #
    # --------------------------------------------- #
    # NaNs and data to be ignored are FALSE.
    specCube = SpectralCube(contSubCube,pacsWcs,mask=cubeMask,fill_value=1.)

    # For convenience, convert the X-axis to km/s
    # (WCSLIB automatically converts to m/s even if you give it km/s)
    specCube = specCube.with_spectral_unit(u.km/u.s)


    # --------------------------------------------- #
    # Build the pySpecCube using the spectral cube. #
    # --------------------------------------------- #
    # NaN values are FALSE in the mask.
    pyCube = pyspeckit.Cube(cube=specCube,maskmap=falseMask['mask'][0,:,:])


    # -------------------------------- #
    # Set up for line profile fitting. #
    # -------------------------------- #

    # Build the guesses array
    gc = buildGuessCube((nCols,nRows),
                        contSubCube = contSubCube,
                        vels = vels,
                        validPixels = validPixels,
                        objectName = obsInfo.objectName,
                        minIdx = obsInfo.minProfIdx,
                        maxIdx = obsInfo.maxProfIdx)
    guessCube,gaussAmp = gc[0],gc[1]



    # ----------------------- #
    # Set fitting parameters. #
    # ----------------------- #
    T,F = True,False
    # Mask zeros in the gaussian amplitude array.
    gaussAmpMasked = np.ma.masked_equal(gaussAmp, 0.0, copy=False)
    # min/max values for the parameters.
    minPars = [np.nanmin(gaussAmpMasked)*0.2,obsInfo.profileMin,0]*nComps
    maxPars = [0,obsInfo.profileMax,150]*nComps
    # Enforce the min/max parameter limits?
    minLimits = [T,T,T]*nComps
    maxLimits = [F,T,T]*nComps

    # Send the guesses to pyCube
    pyCube.parcube = guessCube


    # ---------------------#
    # Finally, do the fit! #
    # ---------------------#
    pyCube = parallelFit(contSubCube=contSubCube,
                         pyCube=pyCube,
                         vels=vels,
                         profileMin=obsInfo.profileMin,
                         profileMax=obsInfo.profileMax,
                         validPixels=validPixels,
                         nComps=nComps)


    # -------------------------------------------------------------- #
    # Compute the discrete fluxes of the fitted gaussian components. #
    # -------------------------------------------------------------- #
    comp1 = calcGaussFluxes(gaussParams = pyCube.parcube[0:3,:,:],
                            validPixels = validPixels,
                            velMin = obsInfo.velMin,
                            velMax = obsInfo.velMax)
    comp2 = calcGaussFluxes(gaussParams = pyCube.parcube[3:6,:,:],
                            validPixels = validPixels,
                            velMin = obsInfo.velMin,
                            velMax = obsInfo.velMax)
    if obsInfo.objectName == 'ngc1068':
        comp3 = calcGaussFluxes(gaussParams = pyCube.parcube[6:9,:,:],
                                validPixels = validPixels,
                                velMin = obsInfo.velMin,
                                velMax = obsInfo.velMax)
    else:
        comp3 = None


    # ------------------------------------------------------------- #
    # Compute the total integrated flux of the fitted line profile. #
    # ------------------------------------------------------------- #
    integralMap = calcTotIntFluxes(gaussParams = pyCube.parcube,
                                   validPixels = validPixels,
                                   velMin = obsInfo.velMin,
                                   velMax = obsInfo.velMax)
    # Mask the updated integral map.
    fluxMasked = np.ma.masked_array(integralMap,~pyCube.maskmap)


    # -------------------------------------------------- #
    # Compute the velocities of the fitted line profile. #
    # -------------------------------------------------- #
    velDict = {'v002':{'frac':2e-3,'image':[]},
               'v02':{'frac':0.023,'image':[]},
               'v16':{'frac':0.156,'image':[]},
               'v50':{'frac':0.500,'image':[]},
               'v84':{'frac':0.841,'image':[]},
               'v97':{'frac':0.977,'image':[]},
               'v99':{'frac':0.998,'image':[]}
              }
    for key in velDict:
        velDict[key]['image'] = calcVel(fluxes = fluxMasked,
                                        velMin = obsInfo.velMin,
                                        velMax = obsInfo.velMax,
                                        frac = velDict[key]['frac'],
                                        validPixels = validPixels,
                                        gaussParam = pyCube.parcube)
        # Do some masking.
        velDict[key]['image'] = np.ma.masked_array(velDict[key]['image'],
                                                   ~pyCube.maskmap)

    # Compute the widths at 1 and 2 sigma and the width of the asymmetry.
    widths = calcVelWidth(velDict)
    w1,w2,wAsym = widths[0], widths[1], widths[2]


    # -------------------------------- #
    # Correct the 1 and 2 sigma widths #
    # for the instrumental resolution. #
    # -------------------------------- #
    instrSigma = lineDict[lineName]['specRes'] / 2.355

    w1Corr = np.sqrt(np.square(w1)-np.square(instrSigma))
    w2Corr = np.sqrt(np.square(w2)-np.square(instrSigma * 2.))


    # ------------------------------------------------------ #
    # Make a dictionary of computed line profile properties. #
    # ------------------------------------------------------ #
    propertyDict = {'flux': {'image':fluxMasked,
                             'prefix':'flux',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 1},
                    'v16':  {'image':velDict['v16']['image'],
                             'prefix':'v16',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 2},
                    'v50':  {'image':velDict['v50']['image'],
                             'prefix':'v50',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 3},
                    'v84':  {'image':velDict['v84']['image'],
                             'prefix':'v84',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 4},
                    'wAsym':{'image':wAsym,
                             'prefix':'wAsym',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 5},
                    'w1':   {'image':w1Corr,
                             'prefix':'w1',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 6},
                    'w2':   {'image':w2Corr,
                             'prefix':'w2',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 7},
                    }


    # ------------------------------------------ #
    # Append line profile properties to the FITS #
    # HDU list and save to a new FITS file.      #
    # ------------------------------------------ #
    # pyFits doesn't like masked arrays, so fill the masked elements with 0.
    for key in propertyDict:
        createHdu(np.ma.filled(propertyDict[key]['image'],0),
                  fitsHdu,propertyDict[key]['prefix'])
    # Append an image HDU of the fitted gaussian parameters
    createHdu(pyCube.parcube,fitsHdu,'gaussParam')
    # Append an image HDU of the continuum-subtracted fluxes.
    createHdu(np.ma.filled(contSubCube,0),fitsHdu, 'contSubFluxes')
    # Append an image HDU of fitted continuum fluxes.
    createHdu(np.ma.filled(contFluxesMasked,0),fitsHdu, 'fittedContFluxes')


    # Save the HDU updates to a new file.
    fitsHdu.writeto(fitsSavePath + objectNameBase + '.fits',overwrite=True)


    # -------------------------- #
    # Plot the fitted continuum. #
    # -------------------------- #
    contFitPdfName = contSavePath + objectNameBase + '_contFit.pdf'
    plotSpectra(data = data,
                contFlux = contFluxesMasked,
                maskedData = profileMask['maskedData'],
                xarr = vels,
                velMin = obsInfo.velMin,
                velMax = obsInfo.velMax,
                validPixels = validPixels,
                objectInfo = objectLabel,
                saveFile= contFitPdfName,
                nComps=nComps)


    # --------------------------------------- #
    # Plot the fitted gaussian components and #
    # the continuum-subtracted spectrum.      #
    # --------------------------------------- #
    contSubPdfName = contSavePath + objectNameBase + '_contSub.pdf'
    plotSpectra(data = contSubCube,
                xarr= vels,
                velMin = obsInfo.velMin,
                velMax = obsInfo.velMax,
                gauss1 = comp1,
                gauss2 = comp2,
                gauss3 = comp3,
                validPixels = validPixels,
                v16 = velDict['v16']['image'],
                v50 = velDict['v50']['image'],
                v84 = velDict['v84']['image'],
                objectInfo = objectLabel,
                saveFile = contSubPdfName,
                nComps = nComps)


    # ---------------------------------------- #
    # Make color maps of the corrected images. #
    # ---------------------------------------- #
    w = WCS(obsInfo.originalHdu['image'].header).celestial

    mapPdfName = (mapSavePath + objectNameBase +
                  '_' + propertyDict[key]['prefix'] + '_map.pdf')
    plotMultiMap(propertyDict = propertyDict,
                 objectName = obsInfo.objectName,
                 centerRa = obsInfo.raCenter,
                 centerDec = obsInfo.decCenter,
                 objectInfo = objectLabel,
                 saveFileName = mapPdfName,
                 numColsRows = (4,2),
                 wcs = w,
                 minMax = wcsMinMax)


    print obsInfo.objectName,lineName









