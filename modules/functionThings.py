
'''
    ~~~~~~~~~~~~~~
    myFunctions.py
    ~~~~~~~~~~~~~~
'''

import numpy as np

speedC=299792.458 # speed of light in km/s





def waveToVel(waves,sysWave):
    '''
        Convert wavelength to velocity and correct for redshift.
        sysWave = (1+redshift)*restWavelength
    '''
    return speedC*((waves-sysWave)/sysWave)



def velToWave(vels,restWave,redshift):
    '''
        Convert velocity to wavelength and correct for redshift
    '''
    return restWave*(1+redshift)*((vels/speedC)+1)



def baselineFunc(x,y):
    '''
        Define 3rd, 2nd, and 1st order polynomials
    '''
    if len(y) == 4:
        return y[3] + y[2]*x + y[1]*x**2. + y[0]*x**3.
    if len(y) == 3:
        return y[2] + y[1]*x + y[0]*x**2.
    if len(y) == 2:
        return y[1] + y[0]*x



def gaussFunc(x,y):
    '''
        Define a guassian equation
    '''
    return y[0]*np.exp(-(x-y[1])**2/(2*y[2]**2))


