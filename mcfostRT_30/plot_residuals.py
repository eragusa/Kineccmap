import matplotlib.pyplot as plt
import scipy
import numpy as np
from numpy import random
msun = 1.99e33
gcgs = 6.67e-8
au = 1.5e13
h = 6.6261e-2
kb = 1.38e-16
mjup = 1.898E30
c = 3e10
plt.rc('xtick', labelsize=16.5) 
plt.rc('ytick', labelsize=16.5) 
plt.rcParams.update({'font.size': 14.75})
fontw = {'family' : 'serif','sans-serif': 'Serif'}
Jy =  1E-23
pc = 3.086e+18
sigmasb = 5.67e-5
lsun = 3.9e33
mp = 1.66e-24
from astropy.io import fits
import matplotlib.pyplot as plt
from eddy import rotationmap
import numpy as np


cube12_tot = rotationmap(path=
    #'./12co/30deg/gaussian/12co_30deg_gv0.fits',
    './12co/30deg/quadratic/12co_30deg_v0.fits',
                   uncertainty=
    './12co/30deg/quadratic/12co_30deg_dv0.fits',
                   downsample=0,
                   FOV=50)


cube13_tot = rotationmap(path=
    './13co/30deg/gaussian/13co_30deg_gv0.fits',
                   uncertainty=
    './13co/30deg/gaussian/13co_30deg_dgv0.fits',
                   downsample=1,
                   FOV=50)


P={}
P['inc'] = 30.    # degrees
P['dist'] = 5.   # parsec
#P['r_max'] = 0.5
P['PA'] = 270.
P['vlsr'] = 0.
P['x0'] = 0.0
P['y0'] = -0.00
P['mstar'] = 1.

imshow_resid = {
  "vmin": -1000,
  "vmax": 1000}




cube12_tot.plot_model_residual(params=P,imshow_kwargs=imshow_resid)
plt.show()
#cube12_tot.plot_model_residual(params=P,imshow_kwargs=imshow_resid)
