"""
    Manually select outlier pixels from 2D colormaps of emission line
    profile properties. Find the average value of the outlier's
    surrounding pixels and replace the outlier value with that average.

    This also finds the average along the spectral axis of the 3D cube.
    So each flux on the spectral axis is replaced by
    the average of the corresponding fluxes from neighbors.

"""
import os
import numpy as np
import copy
from astropy.io import fits
from astropy.wcs import WCS

from ObsInfo import ObsInfo
from dictionaryThings import loadDict
from outlierThings import getIndices, averageNeighbors
from fitsThings import createHdu
from pixelThings import findValidPixels, pixMinMax
from plotThings import plotMultiMap

arcsec = '1arc'
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


    # ----------------------------------------- #
    # Create the base names for saving outputs. #
    # ----------------------------------------- #
    # Base for labeling inside plots.
    objectLabel = (obsInfo.objectName.upper()+'  '+str(obsInfo.obsId)+'  '
                   +lineDict[lineName]['texLabel'])
    # Base for the object's file names.
    objectNameBase = (str(obsInfo.obsId)+'_'+obsInfo.objectName
                      +'_'+lineName)
    # Base path to the object's folders.
    objectPathBase = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName
                          +'/'+arcsec+'/')
    if (not os.path.exists(objectPathBase)):os.makedirs(objectPathBase)


    # --------------------------------------- #
    # Open the FITS file that needs cleaning. #
    # --------------------------------------- #
    fitsFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                +arcsec+'/outFitsFiles2/'+objectNameBase+'.fits')
    fitsHdu = fits.open(fitsFile)

    # Build dictionary of line profile properties.
    propDict = {}
    propertyList = ['contSubFluxes','modelFluxes','flux','v50','v84',
                    'w1','w2','wAsym','rms','meanAbsErr']
    for propName in propertyList:
        propDict.setdefault(propName,fitsHdu[propName].data)

    contSubFluxes = propDict['contSubFluxes']
    vels = fitsHdu['zCorrVels'].data
    pacsWcs=WCS(fitsHdu['image'].header).celestial
    nFluxes, nCols, nRows = contSubFluxes.shape
    minProfIdx = (np.abs(vels - (obsInfo.profileMin))).argmin()
    maxProfIdx = (np.abs(vels - (obsInfo.profileMax))).argmin()



    trueMask = np.ma.getmask(np.ma.masked_where(contSubFluxes==0,contSubFluxes))

    validPixels = findValidPixels((nCols,nRows),trueMask[500,:,:])
    wcsMinMax = pixMinMax(validPixels)


    # --------------------------------------------------------- #
    # Compute the average value of pixels surrounding outliers. #
    # --------------------------------------------------------- #
    # propDictCopy = copy.deepcopy(propDict)
    # Iterate outlier spaxel selection until happy :D
    while True:
        fitPrompt = raw_input('Clean the image? (y/n) ')
        if fitPrompt == 'n': break
        if fitPrompt == 'y':

            #propertyImage=raw_input('Which image to view? ')

            propertyImage = np.ma.masked_where(propDict['w1']==0,propDict['w1'])

            # getIndices() returns a dictionary with a list of the outlier
            # coordinates, and a nested dictionary of the outlier coordinates
            # as the key and its neighbor coordinates as a list of tuples.
            outlierDict = getIndices(propertyImage)
            outlierList = outlierDict['outlierList']
            ignoreDict = outlierDict['ignoreDict']

            # Set the pixRadius for neighbor averages.
            pixRadius=raw_input('\nEnter pixRadius to compute averages: \n')
            pixRadius=int(pixRadius)
            print "\nI'm working here woman!\n"

            # Compute the average of the neighbors
            for key in propDict:
                for col,row in outlierList:
                    # Do not include the outlier in the spaxel average.
                    ignoreKey='('+str(col)+','+str(row)+')'
                    ignoreDict[ignoreKey] = (col,row)

                    # Compute the average line profile property value of the neighbors.
                    ave = averageNeighbors(data=propDict[key],
                                           spaxCoords=[col,row],
                                           pixRadius=pixRadius,
                                           ignore=ignoreDict)

                    # Update the spaxel value with the computed average.
                    propDict[key][col,row] = ave

            # Compute the flux averages along the spectral axis.
            for col,row in outlierList:
                    aveSpec = averageNeighbors(data=propDict['modelFluxes'],
                                               spaxCoords=[col,row],
                                               pixRadius=pixRadius,
                                               ignore=ignoreDict,
                                               aveSpecAxis=True)
                    propDict['modelFluxes'][:,col,row] = aveSpec



    propDictCopy = copy.deepcopy(propDict)
    
    # --------------------------------- #
    # Paths to save PDF and FITS files. #
    # --------------------------------- #
    # Line Property Maps
    mapSavePath = objectPathBase + 'propertyMaps/'
    if (not os.path.exists(mapSavePath)):os.makedirs(mapSavePath)
    # Output FITS files
    fitsSavePath = objectPathBase + 'outFitsFiles2/'
    if (not os.path.exists(fitsSavePath)):os.makedirs(fitsSavePath)


    # Make new FITS file with cleaned images and cubes.
    fitsSaveName = fitsSavePath + objectNameBase + '_cleaned.fits'
    for propName in propDictCopy.keys():
        createHdu(propDictCopy[propName],fitsHdu, propName+'Clean')
    createHdu(vels,fitsHdu, 'zCorrVels')

    # Save HDUList to new FITS file.
    fitsHdu.writeto(fitsSaveName, overwrite=True)



    # ---------------------------------------- #
    # Make color maps of the corrected images. #
    # ---------------------------------------- #
    propertyDict0 = {'flux': {'image':propDictCopy['flux'],
                             'prefix':'flux',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 1},
                    'v50':  {'image':propDictCopy['v50'],
                             'prefix':'v50',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 2},
                    'v84':  {'image':propDictCopy['v84'],
                             'prefix':'v84',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 3},
                    'wAsym':{'image':propDictCopy['wAsym'],
                             'prefix':'wAsym',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 4},
                    'w1':   {'image':propDictCopy['w1'],
                             'prefix':'w1',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 5},
                    'w2':   {'image':propDictCopy['w2'],
                             'prefix':'w2',
                             'unit': r'km s$^{-1}$',
                             'subPlotId': 6},
                    'rms':  {'image':propDictCopy['rms'],
                             'prefix':'rms',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 7},
                    'meanAbsErr':  {'image':propDictCopy['meanAbsErr'],
                             'prefix':'meanAbsErr',
                             'unit': r'Jy km s$^{-1}$',
                             'subPlotId': 8}
                   }


    for key in propertyDict0.keys():
        propertyDict0[key]['image'] = np.ma.masked_array(propertyDict0[key]['image'],mask=trueMask[500,:,:])

    # ---------------------------------------- #
    # Make color maps of the corrected images. #
    # ---------------------------------------- #
    mapPdfName = (mapSavePath + objectNameBase + '_propertyMapsClean.pdf')
    plotMultiMap(propertyDict = propertyDict0,
                 objectName = obsInfo.objectName,
                 centerRa = obsInfo.raCenter,
                 centerDec = obsInfo.decCenter,
                 objectInfo = objectLabel,
                 saveFileName = mapPdfName,
                 numColsRows = (4,2),
                 wcs = pacsWcs,
                 minMax = wcsMinMax)


    print obsInfo.objectName, lineName
    break
