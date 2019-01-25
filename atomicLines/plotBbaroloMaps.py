"""
    Comparing colormaps of the data and BBarolo model.

"""
import os
import pickle
import numpy as np
from numpy import ma

from astropy.wcs import WCS
from astropy.io import fits
from scipy.integrate import cumtrapz

from dictionaryThings import loadDict
from ObsInfo import ObsInfo
from pixelThings import findValidPixels, pixMinMax
from maskThings import maskTrue,maskFalse
from fitsThings import buildWcs
from fluxThings import calcGaussFluxes, calcTotIntFluxes
from plotBbaroloMapsThings import plotLineProfiles, makeColormaps
from velocityThings import calcVel



out = 'modOut0'
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


    # ----------------------------------------- #
    # Create the base names for saving outputs. #
    # ----------------------------------------- #
    # Base for the object's file names.
    objectNameBase = (str(obsInfo.obsId)+'_'+obsInfo.objectName
                      +'_'+lineName)
    
    # Base path to the object's folders.
    objectPathBase = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName
                          +'/'+arcsec+'/3dBarolo/')
    if (not os.path.exists(objectPathBase)):os.makedirs(objectPathBase)


    fitsFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                +arcsec+'/outFitsFiles2/'+objectNameBase+'.fits')
    fitsHdu = fits.open(fitsFile)
    vels = fitsHdu['zCorrVels'].data
    data = fitsHdu['contSubFluxes'].data
    trueMask = np.ma.getmask(np.ma.masked_where(data==0,data)) 
    dataMasked = np.ma.masked_array(fitsHdu['contSubFluxes'].data,mask=trueMask)
    dataModMasked = np.ma.masked_array(fitsHdu['modelFluxes'].data,mask=trueMask)
    dataFluxesMasked = np.ma.masked_array(fitsHdu['flux'].data,mask=trueMask[100,:,:])
    dataV50Masked = np.ma.masked_array(fitsHdu['v50'].data,mask=trueMask[100,:,:])
    dataW1Masked = np.ma.masked_array(fitsHdu['w1'].data,mask=trueMask[100,:,:])

    pacsWcs=WCS(fitsHdu['image'].header).celestial
    nRows,nCols,nFluxes = data.shape[2], data.shape[1], data.shape[0]
    minProfIdx = (np.abs(vels - (obsInfo.profileMin))).argmin()
    maxProfIdx = (np.abs(vels - (obsInfo.profileMax))).argmin()
    


    # Make an array of coordinates for the valid spaxels
    validPixels = findValidPixels((nCols,nRows),trueMask[100,:,:])
    wcsMinMax = pixMinMax(validPixels)

    barFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
               +arcsec+'/barModFitting/outFitsFiles/'+objectNameBase+'.fits')
    barHdu = fits.open(barFile)
    barIntFluxes = np.ma.masked_array(barHdu['flux'].data,mask=trueMask[100,:,:])
    barModelFluxes = np.ma.masked_array(barHdu['modelFluxes'].data,mask=trueMask)
    barV50 = np.ma.masked_array(barHdu['v50'].data,mask=trueMask[100,:,:])

    barW1Temp = np.ma.masked_array(barHdu['w1'].data,mask=trueMask[100,:,:])
    barW1 = np.ma.masked_where(barW1Temp==0.,barW1Temp)

    imageListFluxFrac = [dataFluxesMasked, barIntFluxes, dataFluxesMasked-barIntFluxes,
                         dataV50Masked,    barV50, dataV50Masked-barV50,
                         dataW1Masked,     barW1, dataW1Masked-barW1]


    try:
        ringLogFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                       +arcsec+'/3dBarolo/'+lineName+'/'+out+'/ringlog2.txt')
        rad,vrot,disp,inc,pa,z0,xpos,ypos,vsys,vrad = np.genfromtxt(ringLogFile,skip_header=1,usecols=(1,2,3,4,5,7,9,10,11,12),unpack=True) 

    except:
        ringLogFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                       +arcsec+'/3dBarolo/'+lineName+'/'+out+'/ringlog1.txt')
        rad,vrot,disp,inc,pa,z0,xpos,ypos,vsys,vrad = np.genfromtxt(ringLogFile,skip_header=1,usecols=(1,2,3,4,5,7,9,10,11,12),unpack=True) 

    xmin,ymin = 0,0
    xmax,ymax = dataFluxesMasked.shape

    xcen, ycen, phi = [np.nanmean(xpos)-xmin,np.nanmean(ypos)-ymin,np.nanmean(pa)]

    # Coordinates of the major axis.
    x = np.arange(0,xmax-xmin,0.1) 
    y = np.tan(np.radians(phi-90))*(x-xcen)+ycen 

    
    vsys = np.nanmean(vsys)
    pa = np.nanmean(pa)

    radius = np.concatenate((rad,-rad)) 
    vrotation, inclin, vsystem, posang = vrot, inc, vsys, pa  

    vlos1 = vrotation*np.sin(np.deg2rad(inclin))+vsystem 
    vlos2 = vsystem-vrotation*np.sin(np.deg2rad(inclin)) 

    reverse = True 
    if reverse==True: vlos1, vlos2 = vlos2, vlos1 
    vlos = np.concatenate((vlos1,vlos2)) 


    top = topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'+arcsec+'/3dBarolo/'+lineName+'/'+out+'/maps/'+obsInfo.objectName
    mom0 = fits.open(top+'_0mom.fits')[0].data
    mom1 = fits.open(top+'_1mom.fits')[0].data - np.nanmean(vsys)
    mom2 = fits.open(top+'_2mom.fits')[0].data

    mom0Mod = fits.open(top+'_local_0mom.fits')[0].data
    mom1Mod = fits.open(top+'_local_1mom.fits')[0].data - np.nanmean(vsys)
    mom2Mod = fits.open(top+'_local_2mom.fits')[0].data

    imageListMom = [mom0,mom0Mod, mom0-mom0Mod,
                    mom1,mom1Mod, mom1-mom1Mod,
                    mom2,mom2Mod, mom2-mom2Mod]

    newMask = np.ma.getmask(np.ma.masked_invalid(mom2Mod))


    # Line properties computed via fractional flux.
    saveFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                +arcsec+'/3dBarolo/'+lineName+'/'+out+'/'+lineName
                +'_cmapFluxFrac.pdf')

    makeColormaps(images=imageListFluxFrac,
                  wcs=pacsWcs,
                  majX=x,
                  majY=y,
                  xCen=xcen,
                  yCen=ycen,
                  objectName=obsInfo.objectName,
                  lineLabel=lineDict[lineName]['texLabel'],
                  saveFile=saveFile,
                  mask = newMask,
                  savePlot=True)

    # Moments of both data and barolo model.
    saveFile = (topDir+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                +arcsec+'/3dBarolo/'+lineName+'/'+out+'/'+lineName
                +'_cmapMom.pdf')

    makeColormaps(images=imageListMom,
                  wcs=pacsWcs,
                  majX=x,
                  majY=y,
                  xCen=xcen,
                  yCen=ycen,
                  objectName=obsInfo.objectName,
                  lineLabel=lineDict[lineName]['texLabel'],
                  saveFile=saveFile,
                  mask = newMask,
                  savePlot=True)
    
    print obsInfo.objectName, lineName
