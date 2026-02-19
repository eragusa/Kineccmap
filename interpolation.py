from scipy.interpolate import interp1d
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator,griddata,RectBivariateSpline
import pdb

#in this module we define the interpolator, so that if we want then to 
#change it we can change it only here

def interpolator(x,y,opt):
    option=opt #sets how large is window compared to length of the array.
    #first filter with
    wl=int(len(x)/option)
    if wl % 2 == 0:
        wl += 1

    #sistema come option che per sigma serve window stretta perche funzione e' molto steep
    #savgol filter fits a polynomial function of order polyorder across a moving window of length window_length 
    y_filtered=savgol_filter(y,window_length=wl,polyorder=2)
    #then interpolate
    interpolation_method = 'cubic'      
    interp = interp1d(x, y_filtered, kind=interpolation_method,bounds_error=False,fill_value="extrapolate")
    #interp=np.polynomial.Chebyshev.fit(x,y,opt)
    return interp

def interpolator_2D(x,y,z):
   # nx=len(x)
   # ny=len(y)
 #   xgr,ygr=np.meshgrid(x,y,indexing='ij')
  #  xgrplan=xgr.reshape(nx*ny)
   # ygrplan=ygr.reshape(nx*ny)
    #zplan=z.reshape(nx,ny)
 #   pdb.set_trace()
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    z_sorted = z[sorted_indices]
 
    interp_f = RegularGridInterpolator((x_sorted, y_sorted), z_sorted.transpose(),bounds_error=False, fill_value=None)
    return interp_f

def interpolator_2D_nonregular_togrid(x,y,z,xnew,ynew):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    z_sorted = z[sorted_indices]
    interp_f_unstruct=griddata((x_sorted, y_sorted), z_sorted, (xnew, ynew), method='linear',fill_value=np.nan)
    return interp_f_unstruct

def interpolator_2D_spline(x,y,z):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    z_sorted = z[sorted_indices]
    interp_f_spline=RectBivariateSpline(x_sorted, y_sorted, z_sorted, kx=3, ky=3) 
    return interp_f_spline
