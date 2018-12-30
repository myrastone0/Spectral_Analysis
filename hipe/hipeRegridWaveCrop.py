"""
    To be run inside HIPE.

    Retrieve an observation, crop the cube spectrally to eliminate NaNs,
    create an equidistant wavelength grid, and interpolate onto the new grid.
"""
import os

# Conserve the flux when interpolating to smaller sized spaxels?
conserveFlux = True
# Size of the output spaxel in arcsecond.
spaxelSize = 6
# Labeling variable.
arcsec = str(spaxelSize)+'arc'

# Top working directory.
topPath = '/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'

# Read in line by line the parameter file.
paramListName = topPath + 'fittingParametersV4.txt'
paramListFile = open(paramListName,'r')
paramListLines = paramListFile.readlines()
paramListFile.close()


for line in paramListLines:
    col = line.split()
    if not col[0].startswith("#"):  #Skip the header and anything commented out.
        # --------------------------------- #
        # Retrieve info for the observation #
        # from the parameter file.          #
        # --------------------------------- #
        objectName = col[0]
        obsId = int(col[1])
        camera = col[2]
        sliceNum = int(col[3])
        lineName = col[4]


        # -------------------------- #
        # Directories and File Names #
        # -------------------------- #
        # Base for the object's file names.
        objectNameBase = str(obsId)+'_'+objectName+'_'+lineName
        # Path to save images as FITS files
        fitsPath = (topPath+'pySpecKitCube/specInterMixed/'+arcsec +'/')
        if (not os.path.exists(fitsPath)): os.makedirs(fitsPath)


        # ---------------------------- #
        # Retrieve the observation and #
        # the calibration tree.        #
        # ---------------------------- #
        obs = getObservation(obsId,useHsa=1)
        calTree = getCalTree(obs=obs)


        # ----------------------------- #
        # Get the rebinned product cube #
        # corresponding to the camera.  #
        # ----------------------------- #
        if camera == 'blue':
            productCube = 'HPS3DRB'
            # The "slicedCubes" are necessary to build the wavelength grid,
            # the fluxes therein will not be used for anything:
            slicedCubes=obs.refs["level1"].product.refs["HPS3DB"].product
        if camera == 'red':
            productCube = 'HPS3DRR'
            slicedCubes=obs.refs["level1"].product.refs["HPS3DR"].product


        # ------------------------ #
        # Extract the cube context #
        # ------------------------ #
        level = "level2"
        spg =  obs.meta["creator"].string


        #---------------------------- #
        # Get the rebinned cubes and  #
        # crop the NaNs in the fluxes #
        # --------------------------- #
        slicedRebinnedCubes = obs.refs["level2"].product.refs[productCube].product

        lineCube = obs.refs["level2"].product.refs[productCube].product.refs[sliceNum].product
        # Loop through all the spaxels and get the indices of the
        # fluxes which are NaN
        nanIdxs=[]
        for ii in range(5):
            for jj in range(5):
                fluxes = lineCube.getFlux(ii,jj)
                for kk in range(len(fluxes)):
                    if IS_NAN(fluxes[kk]) == True:
                        nanIdxs.append(kk)
        # Find the indices which will bound the wavelengths to keep.
        startIdx = max([x for x in nanIdxs if x < 50])
        endIdx = min([x for x in nanIdxs if x > 50])
        # Crop the cube spectrally
        startWave = lineCube.getWave()[startIdx]
        endWave = lineCube.getWave()[endIdx]
        cubeCropped = pacsExtractSpectralRange(slicedRebinnedCubes,
                                               waveRanges=[[startWave,endWave]])


        # -------------------- #
        # Run specInterpolate  #
        # on the cropped cube. #
        # -------------------- #
        slicedInterpolatedCubes = specInterpolate(cubeCropped,
                                                  outputPixelsize = spaxelSize,
                                                  conserveFlux=conserveFlux)


        # ----------------------------------- #
        # Create the wavelength grid that the #
        # cubes will be interpolated onto.    #
        # ----------------------------------- #
        # Get some parameters needed to run wavelengthGrid
        oversampleWave = slicedInterpolatedCubes.getMeta()["oversample"].value
        upsampleWave = slicedInterpolatedCubes.getMeta()["upsample"].value
        # Create the wavelength grid.
        equidistantWaveGrid = wavelengthGrid(slicedCubes,
                                             oversample=oversampleWave,
                                             upsample = upsampleWave,
                                             calTree = calTree,
                                             regularGrid = True,
                                             fracMinBinSize = 0.35)


        # ----------------------------------- #
        # Create the interpolated cubes with #
        # an equidistant wavelength grid.    #
        # ----------------------------------- #
        slicedEQInterpolatedCubes = specRegridWavelength(slicedInterpolatedCubes,
                                                         equidistantWaveGrid)
        # Note that fracMinBinSize = 0.35 is the recommended value to use so that
        # the new spectra follow the old spectra closely,
        # slicedEQInterpolatedCubes is not a single cube, but a context
        # (a group) of cubes, with >=1 actual cubes in the context.


        # ---------------------------------- #
        # Save the interpolated cube to disk #
        # ---------------------------------- #
        fitsFileSaveName = (fitsPath+objectNameBase+
                            '_Hipe_15_0_3244_eqInter'+ arcsec+'.fits')
        #simpleFitsWriter(interpolatedEQcube,fitsFileSaveName)
        simpleFitsWriter(slicedEQInterpolatedCubes.get(0),
                         fitsFileSaveName)


	print objectName, obsId, lineName


