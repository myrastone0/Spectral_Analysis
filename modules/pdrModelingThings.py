"""
    ~~~~~~~~~~~~
    pdrThings.py
    ~~~~~~~~~~~~
"""
import numpy as np
from numpy import ma

from astropy.io import fits
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.axes_grid import AxesGrid
import pywcsgrid2


# ----------------------------- #
# Universal constants and units #
# ----------------------------- #
c = 2.99792458e8        # m/s, speed of light
h = 6.62606957e-34      # J s, Planck constant
k = 1.3806488e-23       # J/K, Boltzmann constant
pc = 3.08567758e16      # m, parsec
Msun = 1.9891e30        # kg, solar mass
kappa350 = .33          # m2/kg, opacity at 350 microns
beta = 1                # power-law exponent



def greybody(wave,T,M):
    """
    This static function returns the grey body flux(Jy) for the specified
    wavelength or wavelengths (micron), for the given observer distance D(pc),
    power-law exponent beta, opacity kappa at 350 micron (m2/kg),
    dust temperature T(K) and dust mass M(Msun).
    """
    nu = c / (wave*1.e-6)                                   # Hz
    nu350 = c / 350.e-6                                     # Hz
    kappa = kappa350 * (nu/nu350)**beta                     # m2/kg
    Bnu = 2*h*nu**3/ c**2 / (np.exp((h*nu)/(k*T)) - 1)      # W/m2/Hz
    flux = M*Msun * kappa * Bnu / (D*pc)**2                 # W/m2/Hz
    return flux * 1.e26                                     # Jy



def fir(i60,i100):
    """
    Compute the FIR total integrated flux (W/m2) from 42.5um to 122.5um using
    the flux densities(Jy) of the greybody SED at 60um and 100um. (Helou1998)

    Parameters:
    -----------
    i60: float
        Specific intensity at 60 um.
    i100: float
        Specific intensity at 100 um.
    """
    return 1.26e-14*((2.58*i60)+i100)



def plotGBFit(dataFluxPoints=None,dataWavePoints=None,
              dataFluxPointsCompare=None,dataWavePointsCompare=None,
              gbFluxes=None,gbWaves=None,
              nColsRows=None,validPixels=None,
              plotTitle=None,saveName=None):
    """
    Plot the line continumm flux points used to fit the FIR SED.
    Overplot the fitted modified blackbody spectrum (a.ka. greybody SED).

    Parameters:
    -----------
    dataFluxPoints: array_like
        A set continuum fluxes used to fit the greybody.
    dataWavePoints: array_like
        A set of wavelengths which correspond to the fluxes in dataFluxPoints.
    dataFluxPointsCompare: array_like
        A set of continuum fluxes (from a different data set)
        for comparison to dataPoints. dataPointsCompare may or may
        not have been used as additional data to complete the GB fitting.
    dataWavePointsCompare: array_like
        A set of wavelengths which correspond to the fluxes
        in dataFluxPointsCompare.
    gbFluxes: array_like
        The fluxes of the fitted GB.
    gbWaves: array_like
        The list of wavelengths used to compute gbFluxes.
    nColsRows: array_like
        The number of columns and rows to make for the plot grid.
    validPixels: array_like
        List of coordinates of the good pixels.
    plotTitle: str
        Title of the plot.
    saveName: str
        The path and/or name of the file to save the plot.
    """
    matplotlib.rcParams['xtick.labelsize'] = 4
    matplotlib.rcParams['ytick.labelsize'] = 4

    nCols,nRows = nColsRows[0],nColsRows[1]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nCols+1,nRows+1,wspace=0.0,hspace=0.0)

    for coord in validPixels:
        col,row = coord[0],coord[1]
        ax = plt.subplot(gs[col,row])

        # Plot the BB function.
        ax.plot(np.log10(gbWaves[:,col,row]),
                np.log10(gbFluxes[:,col,row]),
                linewidth=0.4, alpha=.8)

        # Plot the continuum data points.
        ax.scatter(np.log10(dataWavePoints[:,col,row]),
                   np.log10(dataFluxPoints[:,col,row]),
                   s=2,color='green')
        if dataFluxPointsCompare is not None:
            ax.scatter(np.log10(dataWavePointsCompare[:,col,row]),
                       np.log10(dataFluxPointsCompare[:,col,row]),
                       s=2,color='red')

        # Hide the tick marks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Set figure title.
    fig.text(0.5, 0.9, plotTitle,ha='center',fontsize=10)

    # Save the plot to PDF
    pp = PdfPages(saveName)
    pp.savefig(fig,bbox_inches='tight')
    pp.close()
    plt.close()



def cMapDustTempMass(data,wcs=None,wcsMinMax=None,fluxImage=None,
                     raCenter=None,decCenter=None,
                     plotTitle=None,saveName=None):
    """
    Make a colormap of the dust temperature and dust mass.

    Parameters
    ----------
    data: array_like
        The 2D array of the dust temperatures (axis0)
        and the dust masses (axis1).
    wcs: WCS
        The WCS of the object.
    wcsMinMax: array_like
        The limits (colMin,colMax,rowMin,rowMax) of the colormap spaxels.
    fluxImage: array_like
        Continuum fluxes to overplot on dust properties.
    ra/decCenter: float
        RA and Dec of the galaxy's center.
    plotTitle: str
        The title of the plot.
    saveName: str
        The path and/or name of the file to save the colormaps.
    """
    matplotlib.rcParams['xtick.labelsize'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 6
    matplotlib.rcParams['axes.labelsize'] = 6

    fig = plt.figure()

    # Grid helper
    grid_helper = pywcsgrid2.GridHelper(wcs=wcs)
    # Setup the grid for plotting.
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 2),
                    axes_pad= (0.35,0.07),
                    cbar_mode='each',
                    cbar_location='right',
                    cbar_pad=0,
                    axes_class=(pywcsgrid2.Axes,dict(grid_helper=grid_helper)),
                    share_all=True)
    for ii in range(2):
        # Get the axis.
        ax = grid[ii]
        # Set background color
        ax.patch.set_facecolor('black')
        # Make the tickmarks black.
        ax.tick_params(axis='both', colors='black',width=0)

        # Create the colormap.
        cmap = matplotlib.cm.rainbow
        cmap.set_bad('black',1.)  #Set masked pixels to be black.

        # Crop and plot the image.
        image = data[ii,:,:]
        #im = ax.contourf(image,cmap=cmap)
        im = ax.imshow(image,cmap=cmap)

        # Label the property inside the subplot.
        if ii == 0 : label = 'Dust Temp [K]'
        if ii == 1 : label = r'Dust Mass [M$_{\odot}$]'
        at = AnchoredText(label, loc=2, prop=dict(size=5))
        ax.add_artist(at)

        # Flip the axis per convention.
        ax.invert_yaxis()

        # Mark the center of the galaxy.
        ax['fk5'].plot(raCenter,decCenter,markeredgewidth=.9,
                       marker='+', color='k',ms=7)

        # Mark flux contours and continuum contours.
        ax.contour(fluxImage,linewidths=0.85,alpha=0.8,colors='black')

        # Make a colorbar.
        cax1 = grid.cbar_axes[ii]
        if ii == 0 : fm = '%d'
        if ii ==1 : fm = '%.1e'
        cbar1 = cax1.colorbar(im,format=fm)
        cbar1.ax.tick_params(labelsize=4)

    # Set figure title.
    fig.text(0.5, 0.82, plotTitle,ha='center',fontsize=10)
    # Save the plot to PDF
    pp = PdfPages(saveName)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    plt.close()



def cMapFirTir(tirValues=None,firValues=None,wcs=None,wcsMinMax=None,
               raCenter=None,decCenter=None,fluxImage=None,
               plotTitle=None,saveName=None):
    """
    Make color maps of the TIR and FIR

    Parameters:
    -----------
    tirValues: array_like
        2D array of the total infrared fluxes.
    firValues: array_like
        2D array of the far infrared fluxes.
    wcs: WCS
        The WCS of the object.
    wcsMinMax: array_like
        The limits (colMin,colMax,rowMin,rowMax) of the colormap spaxels.
    fluxImage: array_like
        The continuum fluxes to overplot contours.
    ra/decCenter: float
        RA and Dec of the galaxy's center.
    plotTitle: str
        The title of the plot.
    saveName: str
        The path and/or name of the file to save the colormaps.
    """
    matplotlib.rcParams['xtick.labelsize'] = 6
    matplotlib.rcParams['ytick.labelsize'] = 6
    matplotlib.rcParams['axes.labelsize'] = 6
    fig = plt.figure()

    # Grid helper
    grid_helper = pywcsgrid2.GridHelper(wcs=wcs)
    # Setup the grid for plotting.
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 2),
                    axes_pad= (0.45,0.07),
                    cbar_mode='each',
                    cbar_location='right',
                    cbar_pad=0,
                    axes_class=(pywcsgrid2.Axes,dict(grid_helper=grid_helper)),
                    share_all=True)
    for ii in range(2):
        # Get the axis.
        ax = grid[ii]
        # Set background color
        ax.patch.set_facecolor('black')

        # Create the colormap.
        cmap = matplotlib.cm.rainbow
        cmap.set_bad('black',1.)  #Set masked pixels to be black.

        # Plot the image.
        if ii == 0:
            image = firValues
            label = r'FIR [W m$^{-2}$]'
        if ii == 1:
            image = tirValues
            label = r'TIR [W m$^{-2}$]'
        #im = ax.contourf(image,cmap=cmap)
        im = ax.imshow(image,cmap=cmap)


        # Label the flux inside the subplot.
        at = AnchoredText(label, loc=2, prop=dict(size=5))
        ax.add_artist(at)
        # Flip the axis per convention.
        ax.invert_yaxis()
        # Mark the center of the galaxy.
        ax['fk5'].plot(raCenter,decCenter,markeredgewidth=.9,
                       marker='+', color='k',ms=7)
        # Mark flux contours and continuum contours.
        ax.contour(fluxImage,linewidths=0.85,alpha=0.8,colors='black')

        # Make the tickmarks black.
        ax.tick_params(axis='both', colors='black',width=0)
        # Make a colorbar for each image.
        cax1 = grid.cbar_axes[ii]
        cbar1 = cax1.colorbar(im,format='%.2e')
        cbar1.ax.tick_params(labelsize=4)
    # Set figure title.
    fig.text(0.5, 0.82, plotTitle,ha='center',fontsize=10)
    # Save the plot to PDF
    pp = PdfPages(saveName)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    plt.close()



def cMapPdrParams4Fluxes(pdrParams=None,
                         wcs=None,wcsMinMax=None,
                         raCenter=None,decCenter=None,
                         objectName=None,
                         plotTitle=None,saveFileName=None,
                         cmapMinMax=None):
    """
    Plot the PDR parameters which have been "corrected" to
    more believable values.

    Parameters:
    -----------
    pdrParams: array_like
        Array containing the fitted PDR Toolbox parameters: (chi2,nH,G0)
    wcs: WCS
        Coordinate information.
    wcsMinMax: array_like
        Min/max pixel values of each property. This
        is used to spatially crop the WCS.
    raCenter/decCenter: float
        The center of the object in degrees.
    objectName: str
        Name of the object.
    plotTitle: str
        Title of the entire plot.
    saveFileName: str
        Pathway and name of file to save the PDF.
    cmapMinMax: array_like
        Min/max values of n and G
    """
    matplotlib.rcParams['xtick.labelsize'] = 3
    matplotlib.rcParams['ytick.labelsize'] = 3
    matplotlib.rcParams['axes.labelsize'] = 3


    # Shape of the 2D spatial array.
    nCols,nRows = pdrParams[0,:,:].shape[0],pdrParams[0,:,:].shape[1]

    fig = plt.figure(figsize=(6,7))
    # Grid helper
    grid_helper = pywcsgrid2.GridHelper(wcs=wcs)
    # Setup the grid for plotting.
    nrows_ncols = (4,3)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=nrows_ncols,
                    axes_pad= (0.35,0.07),
                    cbar_mode='each',
                    cbar_location='right',
                    cbar_pad=0,
                    axes_class=(pywcsgrid2.Axes,dict(grid_helper=grid_helper)),
                    share_all=True)

    for count in range(len(pdrParams)):
        # Get the axis.
        ax = grid[count]

        # Create the colormap.
        cmap = matplotlib.cm.rainbow
        cmap.set_bad('black',1.) # Set masked pixels to be black.

        # Get the slice to plot
        image = pdrParams[count,:,:]

        # Plot the image
        if cmapMinMax != None:
            if count in [1,4,7,10]:
                vMin,vMax = cmapMinMax[0],cmapMinMax[1]
                im = ax.imshow(image,cmap=cmap,vmin=vMin,vmax=vMax)
            elif count in [2,5,8,11]:
                vMin,vMax = cmapMinMax[2],cmapMinMax[3]
                im = ax.imshow(image,cmap=cmap,vmin=vMin,vmax=vMax)
            else:
                im = ax.imshow(image,cmap=cmap)
        else:
            im = ax.imshow(image,cmap=cmap)
        # Label the ratio used inside the subplot.
        labelList=['$X^2_{t-63-145-158}$','$n$','$G_0$',
                   '$X^2_{Corr158}$','$n_{corr158}$','$G_{0,corr158}$',
                   '$X^2_{corr63}$','$n_{corr63}$','$G_{0,corr63}$',
                   '$X^2_{corr158-63}$','$n_{corr158-63}$','$G_{0,corr158-63}$']

        at = AnchoredText(labelList[count], loc=2, prop=dict(size=4))
        ax.add_artist(at)

        # Label the property average
        ave = '{:.2f}'.format(np.nanmean(image))
        med = '{:.2f}'.format(np.nanmedian(image))
        at1 = AnchoredText('mean: '+ave, loc=4, prop=dict(size=3))
        at2 = AnchoredText(med, loc=3, prop=dict(size=3))
        ax.add_artist(at1)
        ax.add_artist(at2)

        # Flip the axis per convention.
        ax.invert_yaxis()

        # Mark the center of the galaxy.
        ax['fk5'].plot(raCenter,decCenter,markeredgewidth=.6,
                       marker='+', color='k',ms=5)

        # Make the tickmarks black.
        ax.tick_params(axis='both', colors='black',width=0)

        # Make a colorbar for each image.
        cax1 = grid.cbar_axes[count]
        if any([cmapMinMax == None, count in [0,3,6,9]]):
            cbTickValues = np.linspace(np.amin(image),np.amax(image),5,endpoint=True).tolist()
        else:
            cbTickValues = np.linspace(vMin,vMax,5,endpoint=True).tolist()
        cbar1 = cax1.colorbar(im,ticks=cbTickValues)#,format='%.2e')

        if count in [0,3,6,9,8,11]:
            cbar1.ax.set_yticklabels([str('{:.2f}'.format(x)) for x in cbTickValues])
        else:
            cbar1.ax.set_yticklabels([str(int(x)) for x in cbTickValues])
        cbar1.ax.tick_params(labelsize=4)

    # Set figure title.
    fig.text(0.5, 0.9,plotTitle,ha='center',fontsize=10)

    # -------------------- #
    # Save the plot to PDF #
    # -------------------- #
    pp = PdfPages(saveFileName)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    plt.close()



def cMapPdrParams4FluxesNoWcs(pdrParams=None,
                              objectName=None,
                              plotTitle=None,saveFileName=None,
                              cmapMinMax=None):

    matplotlib.rcParams['xtick.labelsize'] = 4
    matplotlib.rcParams['ytick.labelsize'] = 4
    matplotlib.rcParams['axes.labelsize'] = 4

    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(4,3,wspace=0.3,hspace=0.1)

    count = 0
    for col in range(4):
        for row in range(3):
            # Get the axis.
            ax = plt.subplot(gs[col,row])

            # Create the colormap.
            cmap = matplotlib.cm.rainbow
            cmap.set_bad('black',1.) # Set masked pixels to be black.

            # Get the slice to plot
            image = pdrParams[count,:,:]
            im = ax.imshow(image,cmap=cmap)
#             # Plot the image
#             if cmapMinMax != None:
#                 if count in [1,4,7,10]:
#                     vMin,vMax = cmapMinMax[0],cmapMinMax[1]
#                     im = ax.imshow(image,cmap=cmap,vmin=vMin,vmax=vMax)
#                 elif count in [2,5,8,11]:
#                     vMin,vMax = cmapMinMax[2],cmapMinMax[3]
#                     im = ax.imshow(image,cmap=cmap,vmin=vMin,vmax=vMax)
#                 else:
#                     im = ax.imshow(image,cmap=cmap)
#             else:
#                 im = ax.imshow(image,cmap=cmap)

            # Label the ratio used inside the subplot.
            labelList=['$X^2_{t-63-145-158}$','$n$','$G_0$',
                       '$X^2_{Corr158}$','$n_{corr158}$','$G_{0,corr158}$',
                       '$X^2_{corr63}$','$n_{corr63}$','$G_{0,corr63}$',
                       '$X^2_{corr158-63}$','$n_{corr158-63}$','$G_{0,corr158-63}$']

            at = AnchoredText(labelList[count], loc=2, prop=dict(size=4))
            ax.add_artist(at)

            # Label the property average
            ave = '{:.2f}'.format(np.nanmean(image))
            med = '{:.2f}'.format(np.nanmedian(image))
            at1 = AnchoredText('mean: '+ave, loc=4, prop=dict(size=3))
            at2 = AnchoredText(med, loc=3, prop=dict(size=3))
            ax.add_artist(at1)
            ax.add_artist(at2)

            # Flip the axis per convention.
            ax.invert_yaxis()

            # Make the tickmarks black.
            ax.tick_params(axis='both', colors='black',width=0)

            cbar = fig.colorbar(im,ax=ax, fraction=0.047, pad=0.04)
            cbar.ax.tick_params(labelsize=4)

            count += 1

    # Set figure title.
    fig.text(0.5, 0.9,plotTitle,ha='center',fontsize=10)

    # -------------------- #
    # Save the plot to PDF #
    # -------------------- #
    pp = PdfPages(saveFileName)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    plt.close()
