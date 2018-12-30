"""
    ~~~~~~~~~~~~~
    plotThings.py
    ~~~~~~~~~~~~~
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.signal import argrelmin
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_grid import AxesGrid
import pywcsgrid2
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnchoredText
from numpy import ma
from cleanImageThings import cleanImage

rc('text',usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['AppleGothic']})




def plotSpectra(data=None,contFlux=None,maskedData=None,xarr=None,
                gauss1=None,gauss2=None,gauss3=None,gauss4=None,
                validPixels=None,
                v50=None,v16=None,v84=None,velMin=None,velMax=None,
                saveFile=None,objectInfo=None, nComps=None,
                dontCrop=None):
    """
        Make a plot grid of the two (or three) gaussian components,
        their sum, and the continuum subtracted data.


        Parameters:
        -----------
        data: np.ndarray
            Spectrum fluxes, either subtracted or not.
        contFlux: np.ndarray
            Flux of the fitted contintuum. If set to None, the fitted continuum
            overplotted on the raw data will not be plotted/saved.
        maskedData: np.ndarray
            2D mask used to mask the edges and the location of the line profile.
            The unmasked regions are the regions used to fit the continuum.
        xarr: np.ndarray
            An ndarray with ndim = 1 of the wavelengths/velocities along
            the spectral axis.
        gauss1/gauss2: np.ndarray
            1D array(s) of the first/second fitted gaussian component fluxes.
        validPixels: array_like
            List of (col,row) values of valid pixels.
        v16,v50,v84: np.ndarray
            Images of the measured velocities.
        velMin/velMax: float
            Minimum and maximum velocities of the spectrum to plot.
        saveFile: str
            Full path and name of the file to save as the PDF plot.
        fluxImage: np.ndarray
            Image of total integrated fluxes. Used to overplot contours.
        nComps = int
            If None, 2 components are plotted. Otherwise, set to 3.

        For labeling purposes, objectInfo is:
        -------------------------------------
        objectName: str
            Name of the object.
        obsId: str
            Observation ID
        lineName: str
            Name of the line to be plotted.


        Return:
        -------
        PDF of plotted line spectra in a grid.
    """
    fig = plt.figure(figsize=(8, 8))

    ## Get the number of fluxes, columns, and rows in the data cube.
    npar = data.shape[0]
    ncol = data.shape[1]
    nrow = data.shape[2]


    ## Create a [nrow X ncol] plot grid with no
    ## space between subplots.
    ## Gridspec index starts at 1 instead of 0.
    gs = gridspec.GridSpec(nrow+1, ncol+1, wspace=0.0, hspace=0.0)

    ## Get the indices of the min/max velocities.
    minIdx = (np.abs(xarr - (velMin))).argmin()
    maxIdx = (np.abs(xarr - (velMax))).argmin()

    ## The OH doesn't need to be cropped for some reason.
    if dontCrop != None:
        velArr=xarr
        velRange=np.arange(velMin,velMax,2)
    else:
        ## Crop the edges of the spectrum which are excluded from all fitting.
        velArr = xarr[int(minIdx):int(maxIdx)]
        velRange = np.arange(velMin,velMax,1)

    for ii in range(nrow):
        for jj in range(ncol):
            if (jj,ii) in validPixels:
                ax = plt.subplot(gs[ii, jj]) # Location of the subplot on the grid.

                ## Plot the spectrum (either subtracted or not).
                if dontCrop != None:
                    ax.plot(velArr, data[:,jj,ii], color='black',
                        linewidth=0.4, alpha=.8, drawstyle='steps-mid')
                else:
                    ax.plot(velArr, data[minIdx:maxIdx,jj,ii], color='black',
                        linewidth=0.4, alpha=.8, drawstyle='steps-mid')

                if gauss1 is not None:
                    ## Plot the fitted gaussians
                    ax.plot(velRange, gauss1[:,jj,ii],color='blue',
                            linewidth=0.2,alpha=.7)
                    ax.plot(velRange, gauss2[:,jj,ii],color='blue',
                            linewidth=0.2,alpha=.7)

                    if nComps ==3:
                      ax.plot(velRange, gauss3[:,jj,ii],color='blue',
                              linewidth=0.2,alpha=.7)

                      ax.plot(velRange,gauss1[:,jj,ii]+\
                            gauss2[:,jj,ii]+\
                            gauss3[:,jj,ii],
                            color='magenta',
                            linewidth=0.2,alpha=.5)

                    if nComps ==4:
                      ax.plot(velRange, gauss3[:,jj,ii],color='blue',
                              linewidth=0.2,alpha=.7)

                      ax.plot(velRange, gauss4[:,jj,ii],color='blue',
                              linewidth=0.2,alpha=.7)

                      ax.plot(velRange,gauss1[:,jj,ii]+\
                            gauss2[:,jj,ii]+\
                            gauss3[:,jj,ii]+\
                            gauss4[:,jj,ii],
                            color='magenta',
                            linewidth=0.2,alpha=.5)


                    else:
                      ax.plot(velRange,gauss1[:,jj,ii]+\
                              gauss2[:,jj,ii],
                              color='magenta',
                              linewidth=0.2,alpha=.5)

                ## Mark the measured velocities with a vertical line.
                if v50 is not None:
                    ax.axvline(v50[jj,ii],linewidth=0.2,color='green',
                               linestyle='dotted',alpha=0.5)

                if v16 is not None:
                    ax.axvline(v16[jj,ii],linewidth=0.2,color='red',
                               linestyle='dotted',alpha=0.5)
                    ax.axvline(v84[jj,ii],linewidth=0.2,color='red',
                               linestyle='dotted',alpha=0.5)

                ## Plot the fitted continuum
                if contFlux is not None:
                    ax.plot(velArr,maskedData[minIdx:maxIdx,jj,ii],
                            color='magenta',linewidth=0.2)
                    ax.plot(velArr,contFlux[minIdx:maxIdx,jj,ii],
                            linestyle='dashed', color='blue',linewidth=0.2)


                ## Hide the tick marks
                ax.set_xticks([])
                ax.set_yticks([])

                ax.set_xlim([velMin,velMax])

                ## Mark velocity = 0.
                ax.axvline(0, color='grey', linewidth=0.3, alpha=0.6)

                ## Label the [spaxCol x spaxRow]. The [col x row] convention for the
                ## data cube is opposite the [row x col] convention of gridspec.
                ax.annotate(str(jj) + 'x' + str(ii), (0.6,0.8), xycoords='axes fraction',
                            color='black', fontsize=2)



    ## Show ticks only on the outside spines
    allAxes = fig.get_axes()

    for ax in allAxes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)


    ## Set title and axis labels.
    #fig.text(0.5, 0.9999, objectInfo, ha='center')
    #fig.text(0.5, 0.0001, r'Velocity [km s$^{-1}$] ', ha='center')
    #fig.text(0.0001, 0.5, r'Flux [Jy] ', va='center', rotation='vertical')
    fig.text(0.5, 0.8, objectInfo, ha='center')
    fig.text(0.5, 0.2, r'Velocity [km s$^{-1}$] ', ha='center')
    fig.text(0.2, 0.5, r'Flux [Jy] ', va='center', rotation='vertical')


    ###################
    ## Save the plot ##
    ###################
    pp = PdfPages(saveFile)
    pp.savefig(fig, bbox_inches='tight')
    pp.close()
    plt.close()




def plotMultiMap(propertyDict=None,objectName=None,
                 centerRa=None,centerDec=None,
                 objectInfo=None, saveFileName=None,
                 numColsRows=(1,1),
                 wcs=None,minMax=None,
                 interpolation=None):

  """
    Make a grid of plotted images and save to PDF.

    Parameters
    ----------
    propertyDict: dictionary
        Dictionary that holds the images as keys to be plotted.

        Example format
        --------------
        propertyDict = {'flux': {'image':[],
                                       'prefix':'flux',
                                       'unit': r'Jy km s$^{-1}$',
                                       'subPlotId':1}}
    objectName: str
        Name of the object.
    wcs: wcs
        Coordinate information
    centerRa/Dec: float
        RA and Dec of object's center.
    objectInfo: str
        String listing object name, obsId, and line name. Plot title.
    saveFileName: str
        File name to save the PDF.
    numColsRows: tuple(int)
        Number of (columns,rows) of the grid plot.
    minMax: array_like
        List of min/max values of the columns and rows of the image.
        This is used to crop the extra space around the image.
        [colMin,colMax,rowMin,rowMax]
    interpolation: str
        Interpolation method for the map.


    """
  matplotlib.rcParams['xtick.major.pad']=5


  plotCol, plotRow= numColsRows[0],numColsRows[1]


  fig = plt.figure()

  for key in propertyDict:
    subPlotId = propertyDict[key]['subPlotId']
    ## Crop the image spatially.
    image = propertyDict[key]['image'][minMax[0]:minMax[1],minMax[2]:minMax[3]]
    unit = propertyDict[key]['unit']
    label = propertyDict[key]['prefix']


    ax=fig.add_subplot(plotRow,plotCol,subPlotId,
                       projection=wcs[minMax[0]:minMax[1],minMax[2]:minMax[3]])


    ## Set the colormap theme.
    cmap = matplotlib.cm.rainbow

    ## Set masked pixels to be black.
    cmap.set_bad('black',1.)

    ## Display the image.
    im = ax.imshow(image,cmap=cmap,interpolation=interpolation)

    ## Format and display the colorbar.
    if any([objectName == 'm82',objectName=='ngc253']):
      cbar = fig.colorbar(im,ax=ax, fraction=0.042, pad=0.04)
    elif objectName == 'ngc6240':
      cbar = fig.colorbar(im,ax=ax, pad=0.04)
    else:
      cbar = fig.colorbar(im,ax=ax, fraction=0.047, pad=0.04)
    cbar.ax.tick_params(labelsize=4)



    ## Mark flux contours
    try:
        fluxImage = propertyDict['flux']['image']
    except:
        fluxImage = propertyDict['totalFlux']['image']
    ax.contour(fluxImage[minMax[0]:minMax[1],minMax[2]:minMax[3]],
               9,linewidths=0.65,alpha=0.9,colors='black')

    try:
        ## Plot and fill the property contours
        contourLevels=np.linspace(np.nanmin(image),
                                  np.nanmax(image),
                                  16,endpoint=True,dtype=int)
        im = ax.contourf(image,
                         contourLevels,cmap=cmap,
                         interpolation='nearest')
    except:
        pass


    ## Mark the center of the galaxy.
    plt.plot(centerRa, centerDec,
             transform=ax.get_transform('fk5'),
             markeredgewidth=.6, marker='+', color='k',ms=5)


    ## Label the line profile property
    ax.annotate(label, (0.05,0.9), xycoords='axes fraction',
                color='white', fontsize=4)

    ## Make the tickmarks black (invisible in this case).
    ax.tick_params(axis='both', colors='black',width=0)

    ## Flip the axis per convention.
    ax.invert_yaxis()



    ## Get the axes
    ra = ax.coords['ra']
    dec = ax.coords['dec']

    if plotCol*plotRow <= 6:
        ## Hide y-axis for all but left-most plots.
        if subPlotId in [2,3,5,6]:
          dec.set_ticklabel_visible(False)

        if subPlotId in [1,4]:
          ## Set the axes label padding
          dec.set_axislabel(r'$\delta_{\mathrm{J}2000}$', minpad=-0.25,fontsize=10)

          ## Set the tick label format
          dec.set_major_formatter('dd:mm:ss')


        ## Hide x-axis for all but the bottom-most plots
        if subPlotId in [5]:
            ra.set_axislabel(r'$\alpha_{\mathrm{J}2000}$', minpad=1.2,fontsize=10)

            ## Set the tick label format
            ra.set_major_formatter('hh:mm:ss')
        else:
            ra.set_ticklabel_visible(False)

    if plotCol*plotRow ==8:
        ## Hide y-axis for all but left-most plots.
        if subPlotId in [2,3,4,6,7,8]:
          dec.set_ticklabel_visible(False)

        if subPlotId in [1,5]:
          ## Set the axes label padding
          dec.set_axislabel(r'$\delta_{\mathrm{J}2000}$', minpad=-0.25,fontsize=10)

          ## Set the tick label format
          dec.set_major_formatter('dd:mm:ss')


        ## Hide x-axis for all but the bottom-most plots
        if subPlotId in [6,7]:
            ra.set_axislabel(r'$\alpha_{\mathrm{J}2000}$', minpad=1.2,fontsize=10)

            ## Set the tick label format
            ra.set_major_formatter('hh:mm:ss')
        else:
            ra.set_ticklabel_visible(False)


    ra.set_ticklabel(size=6)
    dec.set_ticklabel(size=6)


    axes=plt.gca()



  ## Set title and axis labels.
  if objectName == 'ngc6240':
    fig.text(0.5, 0.99,objectInfo,ha='center')
  else:
    fig.text(0.5, 0.9,objectInfo,ha='center')

  ## Set padding between subplots.
  if objectName == 'm82':
    fig.tight_layout(w_pad=0.15, h_pad=-8.5)
  elif objectName == 'ngc6240':
    fig.tight_layout(w_pad=0.15)
  else:
    fig.tight_layout(w_pad=0.15, h_pad=-11.5)


  ##########################
  ## Save the plot to PDF ##
  ##########################
  pp = PdfPages(saveFileName)
  pp.savefig(fig, bbox_inches='tight')
  pp.close()
  plt.close()
