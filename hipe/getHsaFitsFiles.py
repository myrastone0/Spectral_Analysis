"""
    Download the level 2 equidistant interpolated
    (or the rebinned) fits files from HSA.

    These are the 3 arcsecond projections
    (or the 9.4"x9.4" spaxels, if rebinned).

"""
import sys, os

#topDir = '/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'+\
#         'pySpecKitCube/specInterFitsFiles/equidistantInterpolate/'

topDir = '/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'+\
         'pySpecKitCube/'

# paramFile = open('/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'+\
#             'fittingParametersV3.txt','r')
paramFile = open('/Volumes/QbertPrimary/umdResearch/adapProposalNearby/'+\
            'fittingParametersOh.txt','r')
param=paramFile.readlines()[1:]
paramFile.close()



for line in param:
    if line.startswith('#') != True:
        col = line.split()
        objectName=col[0]
        obsId=int(col[1])
        camera=col[2]
        sliceNum=int(col[3])
        lineName=col[4]

        if camera == 'red':
            #cubeClass="HPS3DEQIR"
            cubeClass='HPS3DRR'
        if camera == 'blue':
            #cubeClass="HPS3DEQIB"
            cubeClass='HPS3DRB'

        obs=getObservation(obsId,useHsa=1)


        ## Setup for saving the spectra as FITS files to disk
        hipeVersion = (str(Configuration.getProjectInfo().track) + '_' +\
                       str(Configuration.getProjectInfo().build)).replace(".","_")

        #nameBasis  = str(obsId)+"_"+objectName+"_" + lineName+\
        #            "_Hipe_"+hipeVersion+"_eqInter3arc.fits"

        nameBasis  = str(obsId)+"_"+objectName+"_" + lineName+\
                     "_Hipe_"+hipeVersion+"_rebinned.fits"

        saveFilePath = topDir+ objectName + '/94arc/rebinnedCubes/'
        if (not os.path.exists(saveFilePath)): os.makedirs(saveFilePath)

        simpleFitsWriter(product=obs.refs["level2"].product.refs[cubeClass].
                         product.refs[sliceNum].product,
                         file=saveFilePath+ nameBasis)
