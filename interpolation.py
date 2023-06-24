from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import savgol_filter
#in this module we define the interpolator, so that if we want then to 
#change it we can change it only here

def interpolator(x,y,opt):
    option=opt #in case other variables need to be specified from the line
    #first filter with
    y_filtered=savgol_filter(y,window_length=int(len(x)/7),polyorder=2)
    #then interpolate
    interpolation_method = 'cubic'      
    interp = interp1d(x, y_filtered, kind=interpolation_method)
    #interp=np.polynomial.Chebyshev.fit(x,y,opt)
    return interp


