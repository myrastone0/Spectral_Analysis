"""
    Fits a single gaussian to the model from BBarolo.

    This is a script which utilizes the ObsInfo class.

    Create an instance of ObsInfo using data and parameters from a single Herschel
    observation. 

    Measure line profile properties from the fits.
    
    Correct widths at 1 and 2 sigma for instrumental resolution.

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
from initialGuessThings import buildGuessCube
from fluxThings import calcGaussFluxes, calcTotIntFluxes
from velocityThings import calcVel, calcVelWidth
from lineFittingThings import parallelFit
from dictionaryThings import loadDict
from ObsInfo import ObsInfo

out = 'modOut0'
# Size of the interpolated spaxels
arcsec = '3arc'

topDir = '/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'


# ----------------------------------------- #
# Necessary parameter file and dictionaries #
# ----------------------------------------- #
# Dictionary of galaxy properties
objDictName = topDir + 'objectInfoDict.pkl'
# Dictionary of emission line properties.
lineDict = loadDict(topDir + 'emiLineDict.pkl')
# Text file of line fitting parameters.
paramFileName = topDir + 'fittingParametersV4.txt'


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
    # Number of Gaussian components to fit to the line profile.
    nComps = 1


    # ---------------------------------- #
    # Directory and Labeling Name Bases  #
    # ---------------------------------- #
    # Base for labeling inside plots.
    objectLabel = (obsInfo.objectName.upper()+'  '+str(obsInfo.obsId)+'  '
                   +lineDict[lineName]['texLabel'])
    # Base for the object's file names.
    objectNameBase = (str(obsInfo.obsId)+'_'+obsInfo.objectName+'_'+lineName)
    # Base path to the object's directories.    
    objectPathBase = (topDir + 'pySpecKitCube/run4/' + obsInfo.objectName + '/' + arcsec + '/barModFitting/')
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


    # ---------------------------------------- #
    # Open and read in FITS file of the model. #
    # ---------------------------------------- #
    fitsFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'+arcsec+'/3dBarolo/'
                +lineName+'/'+out+'/'+obsInfo.objectName+'mod_local.fits')
    fitsHdu = fits.open(fitsFile)
    data = fitsHdu[0].data
    
    
    # Get the velocities from the original data file.
    originalFits = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                    +arcsec+'/outFitsFiles2/'+objectNameBase+'.fits')
    originalHdu = fits.open(originalFits)
    vels = originalHdu['zCorrVels'].data


    # Number of rows, columns, and fluxes in the cropped 3D data array.
    nRows,nCols,nFluxes = data.shape[2], data.shape[1], data.shape[0]
    
    minProfIdx = (np.abs(vels - (obsInfo.profileMin))).argmin()
    maxProfIdx = (np.abs(vels - (obsInfo.profileMax))).argmin()

    # ------------- #
    # Build the WCS #
    # ------------- #
    # The spectral axis is corrected for redshift in buildWcs function.
    pacsWcs=buildWcs(file=originalFits, restWave=restWave, z=obsInfo.z)
    

    # ---------------- #
    # Build the masks. #
    # ---------------- #
    # The masked data and their corresponding masks are returned as dictionaries.
    # Mask where NaNs are TRUE
    trueMask = np.ma.masked_where(data==0,data)
    # Mask where NaNs are FALSE
    falseMask = np.ma.masked_where(data!=0,data)
    # Mask the profile range. NaNs are TRUE
    profileMask = maskProfile(data=trueMask,xarr=vels,
                              minProfIdx=minProfIdx,maxProfIdx=maxProfIdx)
    # Build mask with a WCS that is compatible with SpectralCube.
    # NaNs are FALSE.
    cubeMask = BooleanArrayMask(np.ma.getmask(falseMask),pacsWcs)

    
    # if nFluxes > 400:
    #     mid = 200
    # else:
    #     mid = 100
    
    mid=150

    # Make an array of coordinates for the valid spaxels
    validPixels = findValidPixels((nCols,nRows),np.ma.getmask(trueMask[mid,:,:]))
    
    
    # --------------------------------------------------- #
    # Find the (col,row) min/max limits of valid spaxels. #
    # Min/max will be used to crop the data and the WCS.  #
    # --------------------------------------------------- #
    wcsMinMax = pixMinMax(validPixels)
    
    # --------------------------------------------- #
    # Build the spectral cube using the contSubCube #
    # --------------------------------------------- #
    # NaNs and data to be ignored are FALSE.
    specCube = SpectralCube(data,pacsWcs,mask=cubeMask,fill_value=1.)

    # For convenience, convert the X-axis to km/s
    # (WCSLIB automatically converts to m/s even if you give it km/s)
    specCube = specCube.with_spectral_unit(u.km/u.s)
    
    # --------------------------------------------- #
    # Build the pySpecCube using the spectral cube. #
    # --------------------------------------------- #
    # NaN values are FALSE in the mask.
    pyCube = pyspeckit.Cube(cube=specCube,maskmap=np.ma.getmask(falseMask[mid,:,:]))
    
    
    # -------------------------------- #
    # Set up for line profile fitting. #
    # -------------------------------- #

    # Build the guesses array
    gc = buildGuessCube((nCols,nRows),
                        contSubCube = data,
                        vels = vels,
                        validPixels = validPixels,
                        objectName = obsInfo.objectName,
                        minIdx = 0,
                        maxIdx = len(vels),
                        singleComp = True)
    guessCube,gaussAmp = gc[0],gc[1]



    # ----------------------- #
    # Set fitting parameters. #
    # ----------------------- #
    T,F = True,False
    # Mask zeros in the gaussian amplitude array.
    gaussAmpMasked = np.ma.masked_equal(gaussAmp, 0.0, copy=False)
    # min/max values for the parameters.
    minPars = [np.nanmin(gaussAmpMasked)*0.2,obsInfo.velMin,0]*nComps
    maxPars = [0,obsInfo.velMax,150]*nComps
    # Enforce the min/max parameter limits?
    minLimits = [T,T,T]*nComps
    maxLimits = [F,T,T]*nComps
    
    # Send the guesses to pyCube
    pyCube.parcube = guessCube
    
    
    # ---------------------#
    # Finally, do the fit! #
    # ---------------------#
    pyCube = parallelFit(contSubCube=data,
                         pyCube=pyCube,
                         vels=vels,
                         profileMin=obsInfo.velMin,
                         profileMax=obsInfo.velMax,
                         validPixels=validPixels,
                         nComps=nComps)
    
    
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
    
    w1Corr = np.sqrt(np.square(w1)-np.square(instrSigma * 2.))
    w2Corr = np.sqrt(np.square(w2)-np.square(instrSigma * 4.))


    # ----------------------------- #
    # Jankie way to compute errors. #
    # ----------------------------- #
    if nComps == 1:
        comp1 = calcGaussFluxes(gaussParams = pyCube.parcube[0:3,:,:],
                                validPixels = validPixels,
                                velArr = vels)
        comp2 = None
        comp3 = None
        modelFluxes = comp1
        
    if nComps == 2:
        comp1 = calcGaussFluxes(gaussParams = pyCube.parcube[0:3,:,:],
                                validPixels = validPixels,
                                velArr = vels)
        comp2 = calcGaussFluxes(gaussParams = pyCube.parcube[3:6,:,:],
                                validPixels = validPixels,
                                velArr = vels)
        comp3 = None
        modelFluxes = comp1 + comp2
        
    if nComps == 3:
        comp3 = calcGaussFluxes(gaussParams = pyCube.parcube[6:9,:,:],
                                validPixels = validPixels,
                                velArr = vels)
        modelFluxes = comp1 + comp2 + comp3


    res =  data - modelFluxes
    meanAbsErr = np.sum(np.abs(res), axis=0) / len(vels)
    rms = np.sqrt( np.sum(res**2., axis=0) / len(vels) )


    # ------------------------------------------------------ #
    # Make a dictionary of computed line profile properties. #
    # ------------------------------------------------------ #
    propertyDict = {'flux': {'image':fluxMasked,
                             'prefix':'flux',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 1},
                    'v50':  {'image':velDict['v50']['image'],
                             'prefix':'v50',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 2},
                    'v84':  {'image':velDict['v84']['image'],
                             'prefix':'v84',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 3},
                    'wAsym':{'image':wAsym,
                             'prefix':'wAsym',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 4},
                    'w1':   {'image':w1Corr,
                             'prefix':'w1',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 5},
                    'w2':   {'image':w2Corr,
                             'prefix':'w2',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 6},
                    'rms':  {'image':rms,
                             'prefix':'rms',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 7},
                    'meanAbsErr':  {'image':meanAbsErr,
                             'prefix':'meanAbsErr',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 8}
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
    # Append fitted emission line model.
    createHdu(modelFluxes,fitsHdu, 'modelFluxes')
    # Append the cropped redshift corrected velocities.
    createHdu(vels,fitsHdu,'zCorrVels')

    # Save the HDU updates to a new file.
    fitsHdu.writeto(fitsSavePath + objectNameBase + '.fits',overwrite=True)


    if arcsec != '1arc':
        # --------------------------------------- #
        # Plot the fitted gaussian components and #
        # the continuum-subtracted spectrum.      #
        # --------------------------------------- #
        contSubPdfName = contSavePath + objectNameBase + '_contSub.pdf'
        plotSpectra(data = data,
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
    w = WCS(originalHdu['image'].header).celestial

    mapPdfName = (mapSavePath + objectNameBase + '_propertyMaps.pdf')
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
    #break
