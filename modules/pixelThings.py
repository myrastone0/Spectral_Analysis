"""
    ~~~~~~~~~~~~~~
    pixelThings.py
    ~~~~~~~~~~~~~~
"""

import numpy as np



def findValidPixels((nCols,nRows),maskArray=None,allValid=0):
  """
    Make a list of the (col,row) spaxels which have data.
  """
  ## Make 2D arrays of the column and row indices.
  colArr,rowArr = np.indices(np.zeros((nCols,nRows)).shape)

  ## All spaxels are valid in the original 5x5 arrays.
  if all([nCols==5,nRows==5]):
    isValid = np.any(maskArray) & maskArray

  ## Otherwise, for the specInterpolated arrays
  ## Get the inverse maskArray (i.e. convert the True to False)
  else:
    isValid = np.any(maskArray) & ~maskArray

  if allValid !=0:
    isValid = np.any(maskArray) & maskArray

  return zip(colArr[isValid], rowArr[isValid])


def pixMinMax(validPixels):
  """
    Find the (col,row) min/max limits of valid spaxels.
  """

  ## Sort the validPixels by column and get the min/max values.
  colSort = sorted(validPixels,key=lambda element: element[0])
  colMin,colMax = (colSort[0][0],colSort[-1][0])

  ## Sort the validPixels by row and get the min/max values.
  rowSort = sorted(validPixels,key=lambda element: element[1])
  rowMin,rowMax = (rowSort[0][1],rowSort[-1][1])

  return [colMin,colMax+1,rowMin,rowMax+1]
