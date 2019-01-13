"""
    This is a script which utilizes the ObsInfo class.

    Create an instance of ObsInfo using data and parameters from a single Herschel
    observation. Compute the reference velocity for the spectral axis, 
    update FITS header keywords, and save to new FITS file.
    
    Create the params.txt input file for BBarolo.
"""

# Standard
import os
import numpy as np
# Related
from astropy.wcs import WCS
# Local
from ObsInfo import ObsInfo
from dictionaryThings import loadDict
from functionThings import waveToVel


# Spaxel size
arcsec = '1arc'

# Which program will be reading the output FITS file?
program = '3dBarolo'
#program = 'kpvslice'

topPath = '/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'


# ----------------------------------------- #
# Necessary parameter file and dictionaries #
# ----------------------------------------- #
# Dictionary of galaxy properties
objDictName = topPath+'objectInfoDict.pkl'
# Dictionary of emission line properties.
lineDict = loadDict(topPath+'emiLineDict.pkl')
# Text file of line fitting parameters.
paramFileName = topPath + 'fittingParametersV4.txt'


# -------------------------------------------------- #
# Read in the parameter file containing line profile #
# velocity limits and continuum fitting information. #
# -------------------------------------------------- #
paramFileData = np.genfromtxt(paramFileName, dtype=None, autostrip=True,
                              names=True, encoding=None)


for x in range(len(paramFileData)):
    # ------------------------------------ #
    # Get the galaxy and line information. #
    # ------------------------------------ #
    obsInfo = ObsInfo(x, paramFileName, objDictName)
    
    # Name of the emission line.
    lineName = paramFileData['lineNameShort'][x]
    # Rest wavelength of the emission line.
    restWave = lineDict[lineName]['restWave']
    sysWave = (1.+obsInfo.z)*restWave
    # Spatial resolution at line wavelength
    spatRes = lineDict[lineName]['spatRes']


    # ----------------------------------------- #
    # Create the base names for saving outputs. #
    # ----------------------------------------- #
    # Base for the object's file names.
    objectNameBase = (str(obsInfo.obsId)+'_'+obsInfo.objectName
                      +'_'+lineName)
    
    # Base path to the object's folders.
    objectPathBase = (topPath+'pySpecKitCube/run4/'+obsInfo.objectName
                          +'/'+arcsec+'/'+program+'/')
    if (not os.path.exists(objectPathBase)):os.makedirs(objectPathBase)



    # -------------------------------------------------------------- #
    # Find the minimum velocity of the spectral axis.                #
    # This will be set as the reference velocity in the FITS header. #
    # -------------------------------------------------------------- #
    fitsFile = (topPath+'pySpecKitCube/run4/'+obsInfo.objectName+'/'
                +arcsec+'/outFitsFiles/'+objectNameBase+'.fits')

    if program == 'kpvslice':
        velCorr = True
    if program == '3dBarolo':
        velCorr = False

    obsInfo.computeVels(fitsFile, restWave, velCorr=velCorr)


    # ----------------------------------------------------- #
    # Update/add spectral axis keywords in the FITS header. #
    # Also add beamsize information.                        #
    # ----------------------------------------------------- #
    fitsSavePath = objectPathBase + 'inFits/'
    if (not os.path.exists(fitsSavePath)):os.makedirs(fitsSavePath)
    
    fitsSaveName = fitsSavePath+objectNameBase+'_hdrEditVel.fits'
    
    obsInfo.updateHdr(fitsFile, restWave, beamSize=spatRes, saveFile=fitsSaveName)


    if program == '3dBarolo':
        # --------------------------- #
        # Parameters for barolo input #
        # --------------------------- #
        raCenter, decCenter = obsInfo.raCenter, obsInfo.decCenter
        w = WCS(obsInfo.hdr).celestial
        xPos, yPos = w.wcs_world2pix(raCenter, decCenter, 1)
        vSys = int(waveToVel(sysWave, restWave))


        # ------------------------ #
        # Save BBarolo parameters. #
        # ------------------------ #
        paramFilePath = objectPathBase + lineName + '/'
        if (not os.path.exists(paramFilePath)):os.makedirs(paramFilePath)

        outFile = open(paramFilePath + 'params.txt', 'w')
        outFile.write('FITSFILE    ' + fitsSaveName + '\n'+
                      'OUTFOLDER   ' + objectPathBase + lineName + '/\n'+
                      '{0:<10s}'.format('THREADS') + '{0:<8s}'.format('4') +'\n'+
                      '{0:<10s}'.format('BEAMFWHM') + '{0:<8s}'.format(str(spatRes)) +'\n'+
                      '{0:<10s}'.format('GALFIT') + '{0:<8s}'.format('TRUE') +'\n'+
                      '{0:<10s}'.format('NRADII') + '{0:<8s}'.format('5') +'\n'+
                      '{0:<10s}'.format('RADSEP') + '{0:<8s}'.format('4') +'\n'+
                      '{0:<10s}'.format('VSYS') + '{0:<8s}'.format(str(vSys)) +'\n'+
                      '{0:<10s}'.format('XPOS') + '{0:<8s}'.format(str(int(xPos))) +'\n'+
                      '{0:<10s}'.format('YPOS') + '{0:<8s}'.format(str(int(yPos))) +'\n'+
                      '{0:<10s}'.format('VROT') + '{0:<8s}'.format('60') +'\n'+
                      '{0:<10s}'.format('VDISP') + '{0:<8s}'.format('100') +'\n'+
                      '{0:<10s}'.format('INC') + '{0:<8s}'.format(str(obsInfo.inc)) +'\n'+
                      '{0:<10s}'.format('DELTAINC') + '{0:<8s}'.format('15') +'\n'+
                      '{0:<10s}'.format('PA') + '{0:<8s}'.format(str(obsInfo.majorPA)) +'\n'+
                      '{0:<10s}'.format('DELTAPA') + '{0:<8s}'.format('15') +'\n'+
                      '{0:<10s}'.format('Z0') + '{0:<8s}'.format('.5') +'\n'+
                      '{0:<10s}'.format('FREE') + '{0:<8s}'.format('VROT VDISP INC PA') +'\n'+
                      '{0:<10s}'.format('NORM') + '{0:<8s}'.format('LOCAL') +'\n'+
                      '{0:<10s}'.format('MASK') + '{0:<8s}'.format('NONE') +'\n'+
                      '{0:<10s}'.format('LTYPE') + '{0:<8s}'.format('2') +'\n'+
                      '{0:<10s}'.format('FTYPE') + '{0:<8s}'.format('2') +'\n'+
                      '{0:<10s}'.format('DISTANCE') + '{0:<8s}'.format(str(obsInfo.distance)) +'\n'+
                      '{0:<10s}'.format('BWEIGHT') + '{0:<8s}'.format('1') +'\n'+
                      '{0:<10s}'.format('WFUNC') + '{0:<8s}'.format('2') +'\n'+
                      '{0:<10s}'.format('LINEAR') + '{0:<8s}'.format('1.699') +'\n'+  
                      '{0:<10s}'.format('TWOSTAGE') + '{0:<8s}'.format('TRUE') +'\n'+  
                      '\n')
        outFile.close()

    print obsInfo.objectName, lineName
    break