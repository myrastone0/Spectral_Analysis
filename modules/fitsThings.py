"""
    fitsHandling
"""
from astropy.io import fits
from astropy import wcs

from myFunctions import *

speedC=299792.458 # speed of light in km/s



def createHdu(newData,fitsData,extName):
    """
        Create a new HDU, append it to an open HDU list,
        and give the new HDU a name.

        Parameters:
        -----------
        newData: array_like
            Data that is to be appended to an open HDU list.
        fitsData: = HDUList
            An open FITS file to which the new data is to be appended.
        extName: str
            Name of the HDU that is to be appended.

    """
    ## Create an image HDU of the parameters
    newHdu = fits.PrimaryHDU(newData)

    ## Append the parameter HDU to the HDU list
    fitsData.append(newHdu)

    ## Name the HDU
    newHdr = fitsData[-1].header
    newHdr['EXTNAME'] = extName




def buildWcs(file=None, restWave=None, z=None):
    '''
        Read in a fits file and build a WCS.
        Convert wavelengths to velocities.

        Parameters
        ----------
        file: str
            Name of the fits file.
        restWave: float
            Rest wavelength of a line.
        z: float
            Redshift of the object.
    '''
    ## Calculate the wavelength of the line at
    ## systemic velocity.
    sysWave = (1.+z)*restWave

    ## Open the fits file.
    hdu = fits.open(file)

    ## Get the header information of the spectra HDU
    hdr = hdu['image'].header




    ## Get the WCS header info

    crval1 = hdr['CRVAL1']  # First coordinate of reference pixel
    crval2 = hdr['CRVAL2']  # Second coordinate of reference pixel
    cdelt1 = hdr['CDELT1']  # Pixel scale axis 1, unit=Angle
    cdelt2 = hdr['CDELT2']  # Pixel scale axis 2, unit=Angle



    ## Correct the WCS velocity zero point for redshift
    waveZeroPoint = hdr['CRVAL3']
    zCorrVelZeroPoint = waveToVel(waveZeroPoint,sysWave)
    cdelt3 = (hdr['CDELT3']/sysWave)*speedC

    ## Convert the spectral axis from wavelength to velocity.
    w = wcs.WCS(naxis=3)
    w.wcs.ctype = ['RA---TAN','DEC--TAN','VELO']
    w.wcs.cunit = ['deg','deg','km/s']
    w.wcs.crval = [crval1,crval2,zCorrVelZeroPoint]
    w.wcs.crpix = [1.0,  1.0,  1.0]
    w.wcs.cdelt = np.array([cdelt1,cdelt2,cdelt3])

    return w


def buildWcsOh(file=None):
    '''
        Read in an OH fits file and build a WCS.
        Just a hack to get the WCS for the cubes with non-equidistant
        wavelength grids.

        Parameters
        ----------
        file: str
            Name of the fits file.
        restWave: float
            Rest wavelength of a line.
        z: float
            Redshift of the object.
    '''
    ## Open the fits file.
    hdu = fits.open(file)

    ## Get the header information of the spectra HDU
    hdr = hdu['image'].header

    ## Get the WCS header info
    crval1 = hdr['CRVAL1']  # First coordinate of reference pixel
    crval2 = hdr['CRVAL2']  # Second coordinate of reference pixel
    cdelt1 = hdr['CDELT1']  # Pixel scale axis 1, unit=Angle
    cdelt2 = hdr['CDELT2']  # Pixel scale axis 2, unit=Angle

    naxis1 = hdr['NAXIS1']
    naxis2 = hdr['NAXIS2']

    w = wcs.WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN','DEC--TAN']
    w.wcs.cunit = ['deg','deg']
    w.wcs.crval = [crval1,crval2]
    w.wcs.crpix = [1.0,  1.0]
    w.wcs.cdelt = np.array([cdelt1,cdelt2])
    w.wcs.equinox = 2000.0


    return w





