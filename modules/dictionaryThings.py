"""
  dictionaryThings.py
"""
import pickle
from pixelThings import *

def saveDict(dictionary=None,saveName=None):
  """
    Save a dictionary to disk.

    Parameters:
    -----------
    dictionary: dict
      Self-explanatory.
    saveName: str
      Name of the file to save the dictionary.
  """
  f = open(saveName,'wb')
  pickle.dump(dictionary,f)
  f.close()


def loadDict(dictFile=None):
  """
    Load a dictionary from disk.

    Parameters:
    -----------
    dictFile: str
      File name of the saved dictionary.

    Return:
    -------
    Return the loaded dictionary.
  """
  with open(dictFile,'rb') as f:
    data = pickle.load(f)

  f.close()
  return data


def buildConvolvedLineProfDict(lineProfDict=None,
                               lineName=None,
                               keyValuesList=None,
                               extraKeyValuesList=None,
                               contFluxAdd=False):
  """
    Build the line profile dictionary with the convolved fluxes.

    The empty dictionary must be created before calling this function.

    I'm too lazy to generalize this, so this dictionary build is
    specifically for convolveV*.py

    Parameters:
    -----------
    lineProfDict: dictionary
      The empty dictionary which is to be populated.
    lineName: str
      Name of the emission line or what have you.
    keyValuesList: array_like
      List of each key value.
    extraKeyValuesList: array_like
      List of key values for lines that are NOT [CII]158.
    contFluxAdd: bool
      If True, add the continuum fluxes to lineProfDict.

    NOTE:
      The order of the extra/keyValuesList must be in the
      same order as the extra/keyNamesList below.

  """
  ## Open empty sub-dictionary.
  lineProfDict.setdefault(lineName,dict())
  ## List of names for the dictionary keys.
  keyNamesList =['fitsHdu','gaussParams','validPixels','wcsMinMax','pacsWcs',
                 'fluxJyKmS','fluxJyKmSMasked',
                 'fluxWm2','fluxWm2Masked']
  ## Populate the dictionary key values.
  for ii in range(len(keyValuesList)):
    lineProfDict[lineName].setdefault(keyNamesList[ii],keyValuesList[ii])

  ## Fill in extra keys data if the line is NOT CII158
  if all([lineName != 'cii158', extraKeyValuesList!=None]):
    extraKeyNamesList=['convolvedFluxJyKmS','convolvedFluxJyKmSMasked',
                       'convolvedFluxWm2','convolvedFluxWm2Masked']
    for ii in range(len(extraKeyNamesList)):
      lineProfDict[lineName].setdefault(extraKeyNamesList[ii],
                                        extraKeyValuesList[ii])

  ## If these are 1 arcsec pixels, add the continuum fluxes.
  if contFluxAdd==True:
    contFlux=lineProfDict[lineName]['fitsHdu']['contFlux'+str(lineName)].data
    lineProfDict[lineName].setdefault('contFlux',contFlux)




def buildLineProfDict(lineProfDict=None,
                      lineName=None,
                      keyNamesList=None,
                      keyValuesList=None):
  """
    NEEDS WORK

    Build the line profile dictionary (no convolution).

    The empty dictionary must be created before calling this function.

    I'm too lazy to generalize this, so this dictionary build is
    specifically for plotAllLinesV*.py

    Parameters:
    -----------
    lineProfDict: dictionary
      The dictionary which is to be populated.
    lineName: str
      Name of the emission line or what have you.
    keyNames/ValuesList: array_like
      List of each key name/value.

  """
  ## Open empty sub-dictionary.
  lineProfDict.setdefault(lineName,dict())

  ## Populate the dictionary key values.
  for ii in range(len(keyValuesList)):
    lineProfDict[lineName].setdefault(keyNamesList[ii],keyValuesList[ii])


def buildPropMinMaxDict(lineProfDict=None,keyList=None):
  """
    Compute the min/max values of the line profile property
    to set each observation on the same colorbar scale.

    Parameters:
    -----------
    lineProfDict: dictionary
      Dictionary containing the line profile properties.
    keyList: array_like
      List of key values in lineProfDict to compute the min/max values.
  """
  propMinMax={}
  for keyName in keyList:
    tmpMin=[]
    tmpMax=[]
    for lineName in lineProfDict.keys():
      tmpMin.append(np.nanmin(lineProfDict[lineName][keyName].flatten()))
      tmpMax.append(np.nanmax(lineProfDict[lineName][keyName].flatten()))

    cbMin=np.nanmin(tmpMin)
    cbMax=np.nanmax(tmpMax)
    propMinMax.setdefault(keyName,(cbMin,cbMax))

  return propMinMax


def buildAllLineWcsLimits(lineProfDict=None):
  """
    Compute the min/max values of the (col,row) pixels.
    Used to crop all observations to the same spaxel size.

    Parameters:
    -----------
    lineProfDict: dictionary
      Dictionary containing the line profile properties.
  """
  tempAllLineWcsMinMax = [[],[],[],[]]
  for lineName in lineProfDict.keys():
    for ii in range(0,4):
      tempAllLineWcsMinMax[ii].append(lineProfDict[lineName]['wcsMinMax'][ii])

  allLineWcsMinMax = [np.nanmin(tempAllLineWcsMinMax[0]),
                      np.nanmax(tempAllLineWcsMinMax[1]),
                      np.nanmin(tempAllLineWcsMinMax[2]),
                      np.nanmax(tempAllLineWcsMinMax[3])]

  return allLineWcsMinMax

