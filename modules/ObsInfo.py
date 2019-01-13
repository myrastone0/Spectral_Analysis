import numpy as np

from astropy.io import fits

from dictionaryThings import loadDict
from functionThings import waveToVel

speedC=299792.458 # speed of light in km/s

class ObsInfo():
    """
        Object class that holds a galaxy's physical properties
        and the parameters for spectral line fitting for a single
        observation.

        Basically, just trying to get stuff out of an input parameter file.

        Parameters:
        -----------
        lineNum: int
            Line number in the input parameter file from which to pull info.
        paramFile: str
            Name of the text file containing the spectral line fitting properties.
        objectDict: dict
            Dictionary file name (*.pkl) of galaxy properties.

        Attributes:
        -----------
        objectName  :   Galaxy name.
        obsId       :   Observation ID.
        z           :   Galaxy redshift.
        raCenter    :   RA coordinate of the galaxy center in decimal degrees.
        decCenter   :   DEC coordinate of the galaxy center in decimal degrees.
        inc         :   Galaxy inclination in degrees.
        majorPA     :   Position angle of the galaxy's major axis in degrees.
        distance    :   Physical distance to galaxy in Mpc.

        profileMin  :   Minimum velocity limit of the line profile region in km/s.
        profileMax  :   Maximum velocity limit of the line profile region in km/s.
        velMin      :   Minimum velocity limit of the full spectrum in km/s.
        velMax      :   Maximum velocity limit of the full spectrum in km/s.
        polyorder   :   Order of the polynomial to fit the continuum.
    """


    def __init__(self, lineNum, paramFile, objectDict):

        # Read in the parameter file.
        _paramFile = np.genfromtxt(paramFile, dtype=None, autostrip=True,
                                   names=True, encoding=None)

        # Get parameter file variables.
        self.objectName = _paramFile['objectName'][lineNum]
        self.obsId = _paramFile['obsId'][lineNum]

        # Load object information dictionary.
        # objectDict is of the form dictName = {'obj': {key: value}}
        _dictTemp = loadDict(objectDict)
        self.z = _dictTemp[self.objectName]['redshift']
        self.raCenter = _dictTemp[self.objectName]['centerCoo'][0]
        self.decCenter = _dictTemp[self.objectName]['centerCoo'][1]
        self.inc = _dictTemp[self.objectName]['inclination']
        self.majorPA = _dictTemp[self.objectName]['majorPA']
        self.distance = _dictTemp[self.objectName]['distance']


        # ------------------------------- #
        # Get the fitting parameters and  #
        # limits from the parameter file. #
        # ------------------------------- #
        # Continuum fitting boundaries.
        _includes = np.float_(np.array(_paramFile['include'][lineNum].split(',')))
        _excludes = np.float_(np.array(_paramFile['exclude'][lineNum].split(',')))

        # Min/max velocities which bound the line profile region
        self.profileMin = _excludes[1]
        self.profileMax = _excludes[2]
        # Min/max velocities which bound the entire spectrum
        self.velMin = _excludes[0]
        self.velMax = _excludes[3]
        # Order of polynomial to fit to continuum
        self.polyorder = _paramFile['polyorder'][lineNum]



    def computeVels(self, fitsFile, restWave, velCorr=True):
        """
            Find the reference velocity for the spectral axis in a FITS file.

            This computes 2 arrays of velocities. One is redshift-corrected,
            the other is not.

            Compute indices of velocity ranges.

            Parameters:
            -----------
            fitsFile: str
                FITS file of the data.
            restWave: float
                Rest wavelength of the emission line in microns.
            velCorr: bool
                If True, the reference velocity will be redshift-corrected.
        """

        # Read in the FITS file
        self.originalHdu = fits.open(fitsFile)

        # -------------------------------------- #
        # Get/redshift-correct wavelength array. #
        # -------------------------------------- #
        # Wavelengths corrected for redshift
        _wavesFullCorr = np.array((self.originalHdu['wcs-tab'].data[0][0]/(1.+self.z)).flatten())
        # Don't correct the wavelengths for redshift.
        _wavesFull = np.array(self.originalHdu['wcs-tab'].data[0][0].flatten())


        # ------------------- #
        # Compute velocities. #
        # ------------------- #
        # Corrected for redshift
        self.velsCorr = np.array(waveToVel(_wavesFullCorr,restWave).flatten())
        # Not corrected for redshift.
        self.vels = np.array(waveToVel(_wavesFull,restWave).flatten())


        # ------------------------------------------- #
        # Find indices of velocity region boundaries. #
        # ------------------------------------------- #
        # Index of the minimum redshift corrected velocity.
        # This will be the same index used for the uncorrected velocities.
        self.minIdx = (np.abs(self.velsCorr - (self.velMin))).argmin()
        # Also compute the index of the max redshift corrected velocity.
        self.maxIdx = (np.abs(self.velsCorr - (self.velMax))).argmin()

        # Index of the minimum redshift corrected line profile velocity.
        # Index of the maximum redshift corrected line profile velocity.
        self.minProfIdx = (np.abs(self.velsCorr - (self.profileMin))).argmin()
        # Also compute the index of the max redshift corrected velocity.
        self.maxProfIdx = (np.abs(self.velsCorr - (self.profileMax))).argmin()

        # ------------------------------------ #
        # Get the minimunm/reference velocity. #
        # ------------------------------------ #
        if velCorr == True:
            self.velRef = self.velsCorr[self.minIdx]
        if velCorr == False:
            self.velRef = self.vels[self.minIdx]
            #self.velRef = self.originalHdu['zCorrVels'].data[0]


    def cropData():
        """
            Crop the spectral axis of the data cube.
            Also crop the velsCorr array to be the same length as
            the spectral axis.
        """
        self.velsCrop = self.velsCorr[self.minIdx:self.maxIdx]
        self.dataCrop = self.originalHdu['image'].data[self.minIdx:self.maxIdx,:,:]


    def updateHdr(self, fitsFile, restWave, cunit3='KM/S', beamSize=9.5, saveFile=None):
        """
            Convert the data to float32 and replace -9999 values with 0.

            Add/edit FITS header keywords.

            Parameters:
            -----------
            fitsFile: str
                FITS file to edit.
            restWave: float
                Rest wavelength of the emission line in microns.
            cunit: str
                Units of the spectral axis.
            beamSize: float
                Spatial resolution of the instrument in degrees.
            saveFile: str
                Name to save output FITS file.
        """
        self.originalHdu = fits.open(fitsFile)

        # Grab the data and convert to float32
        self.data = self.originalHdu['contSubFluxes'].data.astype('>f4')

        # Convert bad pixel values to zero.
        self.data[self.data == -9999.] = 0

        # Get the header.
        self.hdr = self.originalHdu['image'].header

        # Compute the velocity step size.
        cdelt3 = (self.hdr['CDELT3']/restWave)*speedC

        # Compute the rest frequency
        freq0 = speedC / (restWave * 10**(-9.))
        self.hdr['FREQ0'] = freq0

        # Update the header with the velocity information.
        self.hdr['CTYPE3'] = 'VEL'
        self.hdr['CRVAL3'] = self.velRef
        self.hdr['CDELT3'] = cdelt3
        self.hdr['CUNIT3'] = cunit3

        # Add PACS beam information in units of arcsec.
        self.hdr['BEAMFWHM'] = beamSize

        # Add object name
        self.hdr['OBJECT'] = self.objectName

        # Remove keyword xtension so we can set this as the PrimaryHDU.
        del self.hdr['XTENSION']


        # Create an HDU of the data and the editted header.
        newHdu = fits.PrimaryHDU(self.data, header=self.hdr)
        # Create the HDUList (it's just one HDU)
        hduList = fits.HDUList([newHdu])
        # Save HDUList to new FITS file.
        hduList.writeto(saveFile, overwrite=True)
